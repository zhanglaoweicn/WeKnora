package chatpipeline

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/Tencent/WeKnora/internal/event"
	"github.com/Tencent/WeKnora/internal/models/api"
	"github.com/Tencent/WeKnora/internal/models/chat"
	"github.com/Tencent/WeKnora/internal/types"
	"github.com/Tencent/WeKnora/internal/types/interfaces"
	"github.com/stretchr/testify/require"
)

// syncEventBus is a thread-safe recorder; the stream plugin emits from a
// background goroutine so the test must guard concurrent appends.
type syncEventBus struct {
	mu     sync.Mutex
	events []types.Event
}

func (b *syncEventBus) On(types.EventType, types.EventHandler) {}

func (b *syncEventBus) Emit(_ context.Context, evt types.Event) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.events = append(b.events, evt)
	return nil
}

func (b *syncEventBus) finalAnswerContents() []string {
	b.mu.Lock()
	defer b.mu.Unlock()
	var out []string
	for _, evt := range b.events {
		if evt.Type != types.EventType(event.EventAgentFinalAnswer) {
			continue
		}
		if data, ok := evt.Data.(event.AgentFinalAnswerData); ok {
			out = append(out, data.Content)
		}
	}
	return out
}

func (b *syncEventBus) finalAnswerEvents() []event.AgentFinalAnswerData {
	b.mu.Lock()
	defer b.mu.Unlock()
	var out []event.AgentFinalAnswerData
	for _, evt := range b.events {
		if evt.Type != types.EventType(event.EventAgentFinalAnswer) {
			continue
		}
		if data, ok := evt.Data.(event.AgentFinalAnswerData); ok {
			out = append(out, data)
		}
	}
	return out
}

// openStreamChat returns a buffered channel preloaded with chunks and never
// closes it, so the stream plugin blocks on the channel until ctx is cancelled
// — deterministically exercising the ctx.Done() branch.
type openStreamChat struct {
	chunks      []types.StreamResponse
	closeStream bool
}

func (m *openStreamChat) Chat(context.Context, []chat.Message, *chat.ChatOptions) (*types.ChatResponse, error) {
	return nil, nil
}

func (m *openStreamChat) ChatStream(
	context.Context, []chat.Message, *chat.ChatOptions,
) (<-chan types.StreamResponse, error) {
	ch := make(chan types.StreamResponse, len(m.chunks))
	for _, c := range m.chunks {
		ch <- c
	}
	if m.closeStream {
		close(ch)
	}
	return ch, nil
}

func (m *openStreamChat) GetModelName() string { return "mock" }
func (m *openStreamChat) GetModelID() string   { return "mock" }

// stubModelService only needs GetChatModel; the rest is unused for this test.
type stubModelService struct {
	interfaces.ModelService
	model chat.Chat
}

func (s *stubModelService) GetChatModel(context.Context, string) (chat.Chat, error) {
	return s.model, nil
}

// scriptedStreamChat serves one preloaded chunk list per ChatStream call and
// records the options each call received, so a retry can be asserted on both
// the model output and the request the retry carried.
type scriptedStreamChat struct {
	calls [][]types.StreamResponse
	opts  []*chat.ChatOptions
}

func (m *scriptedStreamChat) Chat(context.Context, []chat.Message, *chat.ChatOptions) (*types.ChatResponse, error) {
	return nil, nil
}

func (m *scriptedStreamChat) ChatStream(
	_ context.Context, _ []chat.Message, opts *chat.ChatOptions,
) (<-chan types.StreamResponse, error) {
	i := len(m.opts)
	m.opts = append(m.opts, opts)
	ch := make(chan types.StreamResponse, len(m.calls[i]))
	for _, c := range m.calls[i] {
		ch <- c
	}
	close(ch)
	return ch, nil
}

func (m *scriptedStreamChat) GetModelName() string { return "mock" }
func (m *scriptedStreamChat) GetModelID() string   { return "mock" }

func truncatedEmptyStream() []types.StreamResponse {
	return []types.StreamResponse{
		{ResponseType: types.ResponseTypeThinking, Content: "planning"},
		{ResponseType: types.ResponseTypeThinking, Done: true},
		{ResponseType: types.ResponseTypeAnswer, Done: true, FinishReason: "length"},
	}
}

// TestStreamDropsIncompleteHandleOnCancel verifies that cancellation cannot
// leak a model-context protocol fragment to the client.
func TestStreamDropsIncompleteHandleOnCancel(t *testing.T) {
	const ref = "resource://AbCdEfGhIjKlMnOpQrStUv"
	bus := &syncEventBus{}
	model := &openStreamChat{chunks: []types.StreamResponse{
		// Ends with a partial alias prefix ("res://0"), so the stream decoder
		// holds it back waiting for the rest that never arrives before cancel.
		{ResponseType: types.ResponseTypeAnswer, Content: "hello res://0"},
	}}

	chatManage := &types.ChatManage{}
	chatManage.SessionID = "sess-cancel"
	chatManage.UserContent = ref // seeds the registry so res://0001 becomes a known alias
	chatManage.EventBus = bus

	ctx, cancel := context.WithCancel(context.Background())
	plugin := &PluginChatCompletionStream{modelService: &stubModelService{model: model}}
	require.Nil(t, plugin.OnEvent(ctx, types.CHAT_COMPLETION_STREAM, chatManage, func() *PluginError { return nil }))

	// Wait until the pre-hold content has been emitted, then cancel.
	require.Eventually(t, func() bool {
		for _, c := range bus.finalAnswerContents() {
			if c == "hello " {
				return true
			}
		}
		return false
	}, 2*time.Second, 5*time.Millisecond)

	cancel()

	// Give the cancellation path time to flush, then assert that only the
	// meaningful prefix was emitted; the incomplete private handle is dropped.
	require.Eventually(t, func() bool { return len(bus.finalAnswerContents()) >= 1 }, 2*time.Second, 5*time.Millisecond)
	time.Sleep(20 * time.Millisecond)
	require.Equal(t, []string{"hello "}, bus.finalAnswerContents())
}

func TestStreamIgnoresDuplicateTerminalAnswer(t *testing.T) {
	bus := &syncEventBus{}
	model := &openStreamChat{closeStream: true, chunks: []types.StreamResponse{
		{ResponseType: types.ResponseTypeAnswer, Content: "hello"},
		{ResponseType: types.ResponseTypeAnswer, Done: true},
		// Some providers repeat the same terminal response when their EOF
		// sentinel is received after finish_reason.
		{ResponseType: types.ResponseTypeAnswer, Done: true},
	}}

	chatManage := &types.ChatManage{}
	chatManage.SessionID = "sess-duplicate-done"
	chatManage.EventBus = bus
	plugin := &PluginChatCompletionStream{modelService: &stubModelService{model: model}}
	require.Nil(t, plugin.OnEvent(context.Background(), types.CHAT_COMPLETION_STREAM, chatManage, func() *PluginError { return nil }))

	require.Eventually(t, func() bool {
		bus.mu.Lock()
		defer bus.mu.Unlock()
		return len(bus.events) == 2
	}, 2*time.Second, 5*time.Millisecond)

	bus.mu.Lock()
	defer bus.mu.Unlock()
	var answerEvents []event.AgentFinalAnswerData
	for _, evt := range bus.events {
		if evt.Type == types.EventType(event.EventAgentFinalAnswer) {
			answerEvents = append(answerEvents, evt.Data.(event.AgentFinalAnswerData))
		}
	}
	require.Equal(t, []event.AgentFinalAnswerData{{Content: "hello"}, {Done: true}}, answerEvents)
}

func TestStreamReportsEmptyLengthTruncation(t *testing.T) {
	bus := &syncEventBus{}
	model := &openStreamChat{closeStream: true, chunks: []types.StreamResponse{
		{ResponseType: types.ResponseTypeThinking, Content: "planning"},
		{ResponseType: types.ResponseTypeThinking, Done: true},
		{ResponseType: types.ResponseTypeAnswer, Done: true, FinishReason: "length"},
	}}

	chatManage := &types.ChatManage{}
	chatManage.SessionID = "sess-empty-length"
	chatManage.EventBus = bus
	plugin := &PluginChatCompletionStream{modelService: &stubModelService{model: model}}
	require.Nil(t, plugin.OnEvent(
		context.Background(), types.CHAT_COMPLETION_STREAM, chatManage,
		func() *PluginError { return nil },
	))

	want := event.AgentFinalAnswerData{
		Content:   EmptyTruncatedAnswerFallback,
		Done:      true,
		Truncated: true,
	}
	require.Eventually(t, func() bool {
		return len(bus.finalAnswerEvents()) == 1
	}, 2*time.Second, 5*time.Millisecond)
	require.Equal(t, []event.AgentFinalAnswerData{want}, bus.finalAnswerEvents())
}

func TestStreamMarksPartialLengthTruncation(t *testing.T) {
	for _, finishReason := range []string{"length", "max_tokens", "max_output_tokens"} {
		t.Run(finishReason, func(t *testing.T) {
			bus := &syncEventBus{}
			model := &openStreamChat{closeStream: true, chunks: []types.StreamResponse{
				{ResponseType: types.ResponseTypeAnswer, Content: "partial answer"},
				{ResponseType: types.ResponseTypeAnswer, Done: true, FinishReason: finishReason},
			}}

			chatManage := &types.ChatManage{}
			chatManage.SessionID = "sess-partial-" + finishReason
			chatManage.EventBus = bus
			plugin := &PluginChatCompletionStream{modelService: &stubModelService{model: model}}
			require.Nil(t, plugin.OnEvent(
				context.Background(), types.CHAT_COMPLETION_STREAM, chatManage,
				func() *PluginError { return nil },
			))

			require.Eventually(t, func() bool {
				return len(bus.finalAnswerEvents()) == 2
			}, 2*time.Second, 5*time.Millisecond)
			events := bus.finalAnswerEvents()
			require.Equal(t, "partial answer", events[0].Content+events[1].Content)
			require.False(t, events[0].Done)
			require.False(t, events[0].Truncated)
			require.True(t, events[1].Done)
			require.True(t, events[1].Truncated)
			require.NotContains(t, events[1].Content, EmptyTruncatedAnswerFallback)
		})
	}
}

func TestStreamLeavesEmptyNaturalStopUnchanged(t *testing.T) {
	bus := &syncEventBus{}
	model := &openStreamChat{closeStream: true, chunks: []types.StreamResponse{
		{ResponseType: types.ResponseTypeAnswer, Done: true, FinishReason: "stop"},
	}}

	chatManage := &types.ChatManage{}
	chatManage.SessionID = "sess-empty-stop"
	chatManage.EventBus = bus
	plugin := &PluginChatCompletionStream{modelService: &stubModelService{model: model}}
	require.Nil(t, plugin.OnEvent(
		context.Background(), types.CHAT_COMPLETION_STREAM, chatManage,
		func() *PluginError { return nil },
	))

	require.Eventually(t, func() bool {
		return len(bus.finalAnswerEvents()) == 1
	}, 2*time.Second, 5*time.Millisecond)
	require.Equal(t, []event.AgentFinalAnswerData{{Done: true}}, bus.finalAnswerEvents())
}

func TestStreamReportsIncompleteAsError(t *testing.T) {
	bus := &syncEventBus{}
	model := &openStreamChat{closeStream: true, chunks: []types.StreamResponse{
		{ResponseType: types.ResponseTypeAnswer, Content: "partial"},
		{ResponseType: types.ResponseTypeAnswer, Done: true, FinishReason: types.FinishReasonIncomplete},
	}}

	chatManage := &types.ChatManage{}
	chatManage.SessionID = "sess-incomplete"
	chatManage.EventBus = bus
	plugin := &PluginChatCompletionStream{modelService: &stubModelService{model: model}}
	require.Nil(t, plugin.OnEvent(
		context.Background(), types.CHAT_COMPLETION_STREAM, chatManage,
		func() *PluginError { return nil },
	))

	errorMessage := func() string {
		bus.mu.Lock()
		defer bus.mu.Unlock()
		for _, evt := range bus.events {
			if data, ok := evt.Data.(event.ErrorData); ok {
				return data.Error
			}
		}
		return ""
	}
	require.Eventually(t, func() bool { return errorMessage() != "" }, 2*time.Second, 5*time.Millisecond)
	require.Equal(t, types.StreamEndedEarlyError, errorMessage())
	for _, answer := range bus.finalAnswerEvents() {
		require.False(t, answer.Done, "a cut-off stream must not close the answer as complete")
	}
}

// TestStreamRetriesEmptyLengthTruncation verifies that a budget exhausted
// inside thinking with no answer text is retried once with a raised,
// thinking-free budget instead of surfacing the fallback immediately.
func TestStreamRetriesEmptyLengthTruncation(t *testing.T) {
	bus := &syncEventBus{}
	model := &scriptedStreamChat{calls: [][]types.StreamResponse{
		truncatedEmptyStream(),
		{
			{ResponseType: types.ResponseTypeAnswer, Content: "the real answer"},
			{ResponseType: types.ResponseTypeAnswer, Done: true, FinishReason: "stop"},
		},
	}}

	chatManage := &types.ChatManage{}
	chatManage.SessionID = "sess-empty-length-retry"
	chatManage.EventBus = bus
	plugin := &PluginChatCompletionStream{modelService: &stubModelService{model: model}}
	require.Nil(t, plugin.OnEvent(
		context.Background(), types.CHAT_COMPLETION_STREAM, chatManage,
		func() *PluginError { return nil },
	))

	require.Eventually(t, func() bool { return len(model.opts) == 2 }, 2*time.Second, 5*time.Millisecond)
	// The retry must drop thinking and raise the budget to the floor.
	retryOpt := model.opts[1]
	require.Equal(t, emptyAnswerRetryFloorTokens, retryOpt.CompletionBudget())
	require.Equal(t, api.ReasoningOff, retryOpt.ReasoningEffort)
	require.NotNil(t, retryOpt.Thinking)
	require.False(t, *retryOpt.Thinking)

	require.Eventually(t, func() bool { return len(bus.finalAnswerEvents()) >= 1 }, 2*time.Second, 5*time.Millisecond)
	// The answer may arrive split across chunks (the stream decoder bridges
	// partial reference prefixes), so assert on the joined content.
	answers := bus.finalAnswerEvents()
	var joined string
	for _, a := range answers {
		joined += a.Content
	}
	require.Equal(t, "the real answer", joined)
	require.True(t, answers[len(answers)-1].Done)
	require.False(t, answers[len(answers)-1].Truncated)
	require.NotContains(t, joined, EmptyTruncatedAnswerFallback)
}

// TestStreamRetryStillTruncatedFallsBack verifies the fallback is emitted when
// even the raised, thinking-free retry ends truncated with no answer text.
func TestStreamRetryStillTruncatedFallsBack(t *testing.T) {
	bus := &syncEventBus{}
	model := &scriptedStreamChat{calls: [][]types.StreamResponse{
		truncatedEmptyStream(),
		truncatedEmptyStream(),
	}}

	chatManage := &types.ChatManage{}
	chatManage.SessionID = "sess-empty-length-retry-fallback"
	chatManage.EventBus = bus
	plugin := &PluginChatCompletionStream{modelService: &stubModelService{model: model}}
	require.Nil(t, plugin.OnEvent(
		context.Background(), types.CHAT_COMPLETION_STREAM, chatManage,
		func() *PluginError { return nil },
	))

	require.Eventually(t, func() bool { return len(model.opts) == 2 }, 2*time.Second, 5*time.Millisecond)
	require.Eventually(t, func() bool { return len(bus.finalAnswerEvents()) == 1 }, 2*time.Second, 5*time.Millisecond)
	answers := bus.finalAnswerEvents()
	require.Equal(t, EmptyTruncatedAnswerFallback, answers[0].Content)
	require.True(t, answers[0].Done)
	require.True(t, answers[0].Truncated)
}

// TestPrepareChatModelLiftsThinkingBudgetFloor verifies a small configured
// budget is raised when the caller explicitly asked for thinking on, and left
// untouched otherwise.
func TestPrepareChatModelLiftsThinkingBudgetFloor(t *testing.T) {
	thinkingOn := true
	modelService := &stubModelService{model: &openStreamChat{}}

	chatManage := &types.ChatManage{}
	chatManage.SummaryConfig.MaxCompletionTokens = 1024
	chatManage.SummaryConfig.Thinking = &thinkingOn

	_, opt, err := prepareChatModel(context.Background(), modelService, chatManage)
	require.Nil(t, err)
	require.Equal(t, thinkingCompletionFloorTokens, opt.CompletionBudget())

	chatManage.SummaryConfig.Thinking = nil
	_, opt, err = prepareChatModel(context.Background(), modelService, chatManage)
	require.Nil(t, err)
	require.Equal(t, 1024, opt.CompletionBudget())
}
