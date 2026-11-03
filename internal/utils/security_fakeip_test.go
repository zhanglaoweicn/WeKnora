package utils

import (
	"net"
	"strings"
	"testing"
)

// TestCheckResolvedIP_FakeIP covers the SSRF_ALLOW_FAKEIP switch on the
// 198.18.0.0/15 benchmark range — the Fake-IP answer range of Clash-class
// proxies. Everything else (link-local metadata, LAN, loopback) must stay
// blocked regardless of the switch.
//
// checkResolvedIP follows isRestrictedIP semantics: (restricted, reason).
func TestCheckResolvedIP_FakeIP(t *testing.T) {
	fakeIP := net.ParseIP("198.18.0.5")
	if fakeIP == nil {
		t.Fatal("failed to parse fake IP")
	}

	// Default (variable unset): blocked, with the dedicated guidance reason.
	if restricted, reason := checkResolvedIP(fakeIP); !restricted {
		t.Fatalf("fake IP must be blocked by default")
	} else if !strings.Contains(reason, "198.18.0.0/15") || !strings.Contains(reason, "Fake-IP") {
		t.Fatalf("default rejection must carry the Fake-IP guidance, got %q", reason)
	}

	// Opt-in: allowed.
	t.Setenv("SSRF_ALLOW_FAKEIP", "true")
	if restricted, reason := checkResolvedIP(fakeIP); restricted {
		t.Fatalf("SSRF_ALLOW_FAKEIP=true must allow the fake range, got reason %q", reason)
	}

	// Explicit opt-out: blocked again.
	t.Setenv("SSRF_ALLOW_FAKEIP", "false")
	if restricted, _ := checkResolvedIP(fakeIP); !restricted {
		t.Fatalf("SSRF_ALLOW_FAKEIP=false must keep blocking the fake range")
	}

	// Loose truthy values follow the SSRF_DNS_WHITELIST_ONLY parsing style.
	t.Setenv("SSRF_ALLOW_FAKEIP", "1")
	if restricted, _ := checkResolvedIP(fakeIP); restricted {
		t.Fatalf("SSRF_ALLOW_FAKEIP=1 must allow the fake range")
	}

	// The switch must not weaken any other restricted class.
	for _, tc := range []struct {
		ip   string
		what string
	}{
		{"169.254.169.254", "link-local metadata"},
		{"192.168.1.10", "private LAN"},
		{"127.0.0.1", "loopback"},
		{"10.0.0.7", "private"},
		{"100.64.0.1", "CGNAT"},
		{"203.0.113.9", "documentation"},
	} {
		if restricted, _ := checkResolvedIP(net.ParseIP(tc.ip)); !restricted {
			t.Fatalf("%s (%s) must stay blocked even with the fake-IP switch on", tc.what, tc.ip)
		}
	}

	// Public addresses remain allowed with the switch on.
	if restricted, reason := checkResolvedIP(net.ParseIP("8.8.8.8")); restricted {
		t.Fatalf("public IP must stay allowed, got reason %q", reason)
	}
}

// TestIsPublicIP_FakeIP ensures the DNS-pinning helper used by the web
// fetcher follows the same switch.
func TestIsPublicIP_FakeIP(t *testing.T) {
	ip := net.ParseIP("198.18.0.139")
	if IsPublicIP(ip) {
		t.Fatal("fake IP must not count as public by default")
	}
	t.Setenv("SSRF_ALLOW_FAKEIP", "true")
	if !IsPublicIP(ip) {
		t.Fatal("fake IP must count as public when SSRF_ALLOW_FAKEIP=true")
	}
}
