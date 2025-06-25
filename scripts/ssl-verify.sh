#!/bin/bash

# PoUW SSL Certificate Verification Script
# Comprehensive SSL/TLS configuration verification and security testing

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

# Function to load configuration from Python config system
load_config_from_python() {
    if command -v python3 &> /dev/null && [ -f "../config.py" ]; then
        info "Loading SSL configuration from config.py..."
        
        # Extract SSL configuration variables
        eval $(python3 -c "
import sys
sys.path.insert(0, '..')
try:
    from config import get_config_manager
    config = get_config_manager(environment='production').get_config()
    print(f'DOMAIN=\"{config.deployment.domain}\"')
    print(f'ENABLE_SSL=\"{str(config.deployment.enable_ssl).lower()}\"')
    print(f'SSL_STAGING=\"{str(config.deployment.ssl_staging).lower()}\"')
except Exception as e:
    print(f'# SSL config loading failed: {e}', file=sys.stderr)
" 2>/dev/null)
        
        if [ "$ENABLE_SSL" = "true" ] && [ -n "$DOMAIN" ] && [ "$DOMAIN" != "localhost" ]; then
            info "SSL enabled in configuration for domain: $DOMAIN"
        fi
    fi
}

# Configuration
DOMAIN=""
PORT="443"
TIMEOUT="10"
VERBOSE=false

# Load configuration before parsing arguments
load_config_from_python

# Function to log messages
log() {
    echo -e "${GREEN}✓ $1${NC}"
}

warn() {
    echo -e "${YELLOW}⚠ WARNING: $1${NC}"
}

error() {
    echo -e "${RED}✗ ERROR: $1${NC}"
}

info() {
    echo -e "${BLUE}ℹ INFO: $1${NC}"
}

success() {
    echo -e "${GREEN}🎉 $1${NC}"
}

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -d, --domain DOMAIN     Domain to verify (required)"
    echo "  -p, --port PORT         Port to test (default: 443)"
    echo "  -t, --timeout SECONDS   Connection timeout (default: 10)"
    echo "  -v, --verbose           Enable verbose output"
    echo "  --help                  Display this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -d example.com"
    echo "  $0 -d api.pouw.network -p 443 -v"
}

# Function to check prerequisites
check_prerequisites() {
    local missing_tools=()
    
    if ! command -v openssl &> /dev/null; then
        missing_tools+=("openssl")
    fi
    
    if ! command -v curl &> /dev/null; then
        missing_tools+=("curl")
    fi
    
    if ! command -v dig &> /dev/null; then
        missing_tools+=("dig")
    fi
    
    if [ ${#missing_tools[@]} -ne 0 ]; then
        error "Missing required tools: ${missing_tools[*]}"
        echo "Please install missing tools and try again."
        exit 1
    fi
}

# Function to check DNS resolution
check_dns_resolution() {
    info "Checking DNS resolution for $DOMAIN..."
    
    local ipv4_record
    ipv4_record=$(dig +short "$DOMAIN" A @8.8.8.8 2>/dev/null || true)
    
    if [ -n "$ipv4_record" ]; then
        log "IPv4 record: $ipv4_record"
    else
        warn "No IPv4 (A) record found for $DOMAIN"
    fi
    
    local ipv6_record
    ipv6_record=$(dig +short "$DOMAIN" AAAA @8.8.8.8 2>/dev/null || true)
    
    if [ -n "$ipv6_record" ]; then
        log "IPv6 record: $ipv6_record"
    else
        info "No IPv6 (AAAA) record found for $DOMAIN"
    fi
    
    # Check if domain resolves to current server
    local server_ip
    server_ip=$(curl -s --max-time 5 http://checkip.amazonaws.com/ 2>/dev/null || \
                curl -s --max-time 5 http://icanhazip.com/ 2>/dev/null || \
                echo "unknown")
    
    if [ "$ipv4_record" = "$server_ip" ]; then
        log "Domain resolves to current server IP"
    else
        warn "Domain resolves to $ipv4_record but server IP is $server_ip"
    fi
}

# Function to test connectivity
test_connectivity() {
    info "Testing connectivity to $DOMAIN:$PORT..."
    
    if timeout "$TIMEOUT" bash -c "</dev/tcp/$DOMAIN/$PORT" 2>/dev/null; then
        log "TCP connection successful"
    else
        error "Cannot establish TCP connection to $DOMAIN:$PORT"
        return 1
    fi
    
    # Test TLS handshake
    if echo | timeout "$TIMEOUT" openssl s_client -connect "$DOMAIN:$PORT" -servername "$DOMAIN" >/dev/null 2>&1; then
        log "TLS handshake successful"
    else
        error "TLS handshake failed"
        return 1
    fi
}

# Function to check certificate validity
check_certificate_validity() {
    info "Checking certificate validity..."
    
    local cert_info
    cert_info=$(echo | openssl s_client -connect "$DOMAIN:$PORT" -servername "$DOMAIN" 2>/dev/null | \
                openssl x509 -noout -dates -subject -issuer 2>/dev/null)
    
    if [ -z "$cert_info" ]; then
        error "Failed to retrieve certificate information"
        return 1
    fi
    
    # Extract dates
    local not_before not_after
    not_before=$(echo "$cert_info" | grep "notBefore" | cut -d= -f2)
    not_after=$(echo "$cert_info" | grep "notAfter" | cut -d= -f2)
    
    log "Certificate valid from: $not_before"
    log "Certificate valid until: $not_after"
    
    # Check if certificate is currently valid
    local current_time not_before_epoch not_after_epoch
    current_time=$(date +%s)
    not_before_epoch=$(date -d "$not_before" +%s 2>/dev/null || echo 0)
    not_after_epoch=$(date -d "$not_after" +%s 2>/dev/null || echo 0)
    
    if [ "$current_time" -lt "$not_before_epoch" ]; then
        error "Certificate is not yet valid"
        return 1
    elif [ "$current_time" -gt "$not_after_epoch" ]; then
        error "Certificate has expired"
        return 1
    else
        local days_until_expiry
        days_until_expiry=$(( (not_after_epoch - current_time) / 86400 ))
        log "Certificate is valid (expires in $days_until_expiry days)"
        
        if [ "$days_until_expiry" -lt 30 ]; then
            warn "Certificate expires in less than 30 days"
        fi
    fi
    
    # Check certificate subject
    local subject
    subject=$(echo "$cert_info" | grep "subject" | cut -d= -f2-)
    log "Certificate subject: $subject"
    
    # Check certificate issuer
    local issuer
    issuer=$(echo "$cert_info" | grep "issuer" | cut -d= -f2-)
    log "Certificate issuer: $issuer"
}

# Function to verify certificate chain
verify_certificate_chain() {
    info "Verifying certificate chain..."
    
    if echo | openssl s_client -connect "$DOMAIN:$PORT" -servername "$DOMAIN" -verify_return_error >/dev/null 2>&1; then
        log "Certificate chain verification passed"
    else
        warn "Certificate chain verification failed"
        
        # Try to get more details
        local verify_output
        verify_output=$(echo | openssl s_client -connect "$DOMAIN:$PORT" -servername "$DOMAIN" 2>&1 | grep -E "(verify error|Verification error)")
        if [ -n "$verify_output" ]; then
            echo "  Details: $verify_output"
        fi
    fi
}

# Function to check certificate transparency
check_certificate_transparency() {
    info "Checking Certificate Transparency logs..."
    
    # Check for SCT (Signed Certificate Timestamp) in the certificate
    local sct_info
    sct_info=$(echo | openssl s_client -connect "$DOMAIN:$PORT" -servername "$DOMAIN" 2>/dev/null | \
               openssl x509 -noout -text 2>/dev/null | grep -A 5 "CT Precertificate SCTs" || true)
    
    if [ -n "$sct_info" ]; then
        log "Certificate Transparency SCTs found"
        if [ "$VERBOSE" = true ]; then
            echo "$sct_info"
        fi
    else
        info "No Certificate Transparency SCTs found in certificate"
    fi
}

# Function to test TLS configuration
test_tls_configuration() {
    info "Testing TLS configuration..."
    
    # Test supported TLS versions
    local tls_versions=("tls1" "tls1_1" "tls1_2" "tls1_3")
    local supported_versions=()
    
    for version in "${tls_versions[@]}"; do
        if echo | timeout 5 openssl s_client -connect "$DOMAIN:$PORT" -servername "$DOMAIN" "-$version" >/dev/null 2>&1; then
            supported_versions+=("$version")
        fi
    done
    
    if [ ${#supported_versions[@]} -eq 0 ]; then
        error "No supported TLS versions found"
        return 1
    fi
    
    log "Supported TLS versions: ${supported_versions[*]}"
    
    # Check for insecure versions
    if [[ " ${supported_versions[*]} " =~ " tls1 " ]] || [[ " ${supported_versions[*]} " =~ " tls1_1 " ]]; then
        warn "Insecure TLS versions (1.0/1.1) are supported"
    fi
    
    # Check cipher suites
    local cipher_info
    cipher_info=$(echo | openssl s_client -connect "$DOMAIN:$PORT" -servername "$DOMAIN" 2>/dev/null | \
                  grep "Cipher    :" | cut -d: -f2- | xargs)
    
    if [ -n "$cipher_info" ]; then
        log "Active cipher suite: $cipher_info"
        
        # Check for weak ciphers
        if [[ "$cipher_info" =~ "RC4" ]] || [[ "$cipher_info" =~ "DES" ]] || [[ "$cipher_info" =~ "MD5" ]]; then
            warn "Weak cipher suite detected: $cipher_info"
        fi
    fi
}

# Function to check security headers
check_security_headers() {
    info "Checking security headers..."
    
    local response
    response=$(curl -s -I --max-time "$TIMEOUT" "https://$DOMAIN/" 2>/dev/null || true)
    
    if [ -z "$response" ]; then
        warn "Could not retrieve HTTP headers"
        return
    fi
    
    # Check for important security headers
    local headers=(
        "Strict-Transport-Security:HSTS"
        "X-Frame-Options:Frame protection"
        "X-Content-Type-Options:Content type protection"
        "X-XSS-Protection:XSS protection"
        "Content-Security-Policy:CSP"
        "Referrer-Policy:Referrer policy"
    )
    
    for header_check in "${headers[@]}"; do
        local header="${header_check%%:*}"
        local description="${header_check##*:}"
        
        if echo "$response" | grep -qi "^$header:"; then
            local value
            value=$(echo "$response" | grep -i "^$header:" | cut -d: -f2- | xargs)
            log "$description header present: $value"
        else
            warn "$description header missing ($header)"
        fi
    done
}

# Function to test HTTP to HTTPS redirect
test_http_redirect() {
    info "Testing HTTP to HTTPS redirect..."
    
    local http_response
    http_response=$(curl -s -I --max-time "$TIMEOUT" "http://$DOMAIN/" 2>/dev/null || true)
    
    if [ -n "$http_response" ]; then
        if echo "$http_response" | grep -q "Location:.*https://"; then
            log "HTTP to HTTPS redirect is configured"
        else
            warn "HTTP to HTTPS redirect not found"
        fi
    else
        info "HTTP port not accessible or not responding"
    fi
}

# Function to perform overall SSL test
perform_ssl_test() {
    info "Performing comprehensive SSL/TLS test for $DOMAIN..."
    echo
    
    local test_results=()
    
    # Run all tests
    if check_dns_resolution; then
        test_results+=("DNS:PASS")
    else
        test_results+=("DNS:FAIL")
    fi
    
    if test_connectivity; then
        test_results+=("CONNECTIVITY:PASS")
    else
        test_results+=("CONNECTIVITY:FAIL")
        return 1
    fi
    
    if check_certificate_validity; then
        test_results+=("CERTIFICATE:PASS")
    else
        test_results+=("CERTIFICATE:FAIL")
    fi
    
    verify_certificate_chain
    check_certificate_transparency
    test_tls_configuration
    check_security_headers
    test_http_redirect
    
    echo
    success "SSL/TLS verification completed for $DOMAIN"
    
    # Summary
    echo -e "${PURPLE}=== VERIFICATION SUMMARY ===${NC}"
    for result in "${test_results[@]}"; do
        local test="${result%%:*}"
        local status="${result##*:}"
        if [ "$status" = "PASS" ]; then
            echo -e "  ${GREEN}✓${NC} $test"
        else
            echo -e "  ${RED}✗${NC} $test"
        fi
    done
    echo
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--domain)
            DOMAIN="$2"
            shift 2
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -t|--timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Main execution
main() {
    if [ -z "$DOMAIN" ]; then
        error "Domain is required. Use -d or --domain option."
        usage
        exit 1
    fi
    
    check_prerequisites
    perform_ssl_test
}

# Execute main function
main "$@"
