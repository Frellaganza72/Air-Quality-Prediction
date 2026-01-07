"""
================================================================================
API TESTING SCRIPT
Test all backend endpoints
================================================================================
Usage: python test_api.py
"""

import requests
import json
from datetime import datetime, timedelta

BASE_URL = "http://localhost:5000"

def print_section(title):
    """Print formatted section header"""
    print("\n" + "="*80)
    print(f"🧪 {title}")
    print("="*80)

def test_endpoint(name, url, method='GET', data=None, params=None):
    """Test a single endpoint"""
    print(f"\n[TEST] {name}")
    print(f"URL: {url}")
    
    try:
        if method == 'GET':
            response = requests.get(url, params=params, timeout=10)
        elif method == 'POST':
            response = requests.post(url, json=data, timeout=10)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ SUCCESS")
            print(f"Response Preview:")
            print(json.dumps(result, indent=2)[:500] + "...")
            return result
        else:
            print(f"❌ FAILED")
            print(f"Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return None

def main():
    """Run all tests"""
    
    print_section("BACKEND API TESTING")
    
    # Test 1: Home / API Info
    print_section("TEST 1: API Information")
    result = test_endpoint(
        "GET /",
        f"{BASE_URL}/"
    )
    
    # Test 2: Dashboard - Latest Date
    print_section("TEST 2: Dashboard (Latest Date)")
    result = test_endpoint(
        "GET /api/dashboard",
        f"{BASE_URL}/api/dashboard"
    )
    
    if result:
        print(f"\n📊 Dashboard Summary:")
        print(f"  Date: {result.get('date')}")
        print(f"  Status: {result.get('overall_status')}")
        print(f"  PM2.5: {result['pollutants']['pm25']['value']} μg/m³ ({result['pollutants']['pm25']['status']})")
        print(f"  O3: {result['pollutants']['o3']['value']} μg/m³ ({result['pollutants']['o3']['status']})")
        print(f"  CO: {result['pollutants']['co']['value']} μg/m³ ({result['pollutants']['co']['status']})")
    
    # Test 3: Dashboard - Specific Date
    print_section("TEST 3: Dashboard (Specific Date)")
    test_date = "2025-01-15"
    result = test_endpoint(
        f"GET /api/dashboard?date={test_date}",
        f"{BASE_URL}/api/dashboard",
        params={'date': test_date}
    )
    
    # Test 4: Recommendations
    print_section("TEST 4: Recommendations")
    result = test_endpoint(
        "GET /api/recommendations",
        f"{BASE_URL}/api/recommendations"
    )
    
    if result:
        print(f"\n💡 Recommendations Summary:")
        print(f"  ISPU Category: {result.get('ispu_category')}")
        print(f"  Total Categories: {len(result.get('recommendations', {}))}")
        
        # Show first recommendation from each category
        for category, recommendations in result.get('recommendations', {}).items():
            if recommendations:
                print(f"  {category}: {recommendations[0]}")
    
    # Test 5: History - All Data
    print_section("TEST 5: History (All Data)")
    result = test_endpoint(
        "GET /api/history",
        f"{BASE_URL}/api/history"
    )
    
    if result:
        print(f"\n📋 History Summary:")
        print(f"  Total Records: {result.get('total_records')}")
        print(f"  Showing: {len(result.get('data', []))}")
        
        stats = result.get('statistics', {})
        print(f"  Average PM2.5: {stats.get('avg_pm25')} μg/m³")
        print(f"  Days Baik: {stats.get('days_baik')}")
        print(f"  Days Sedang: {stats.get('days_sedang')}")
        print(f"  Days Tidak Sehat: {stats.get('days_tidak_sehat')}")
    
    # Test 6: History - Date Range Filter
    print_section("TEST 6: History (Date Range Filter)")
    result = test_endpoint(
        "GET /api/history with filters",
        f"{BASE_URL}/api/history",
        params={
            'start_date': '2025-01-01',
            'end_date': '2025-01-31'
        }
    )
    
    # Test 7: History - Category Filter
    print_section("TEST 7: History (Category Filter)")
    result = test_endpoint(
        "GET /api/history?category=Tidak Sehat",
        f"{BASE_URL}/api/history",
        params={'category': 'Tidak Sehat'}
    )
    
    if result:
        print(f"\n  Filtered Records: {len(result.get('data', []))}")
    
    # Test 8: Anomalies
    print_section("TEST 8: Detected Anomalies")
    result = test_endpoint(
        "GET /api/anomalies",
        f"{BASE_URL}/api/anomalies",
        params={'limit': 10}
    )
    
    if result:
        print(f"\n⚠️ Anomalies Summary:")
        print(f"  Total Anomalies: {result.get('total_anomalies')}")
        print(f"  Showing: {len(result.get('anomalies', []))}")
        
        if result.get('anomalies'):
            print(f"\n  Recent Anomalies:")
            for i, anomaly in enumerate(result['anomalies'][:3], 1):
                print(f"    {i}. {anomaly['date']}: {anomaly['description']}")
    
    # Summary
    print_section("TEST SUMMARY")
    print("✅ All tests completed!")
    print("\n📌 Next Steps:")
    print("  1. Check response times")
    print("  2. Test error handling (invalid dates, etc)")
    print("  3. Load testing with multiple concurrent requests")
    print("  4. Frontend integration testing")

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║  BACKEND API TESTING TOOL                                    ║
print("    ║  Monitor Kualitas Udara                                      ║")
    ╚═══════════════════════════════════════════════════════════════╝
    
    📋 Prerequisites:
       1. Backend server running: python app.py
       2. Models loaded in /models directory
       3. Data available in /data directory
    
    🚀 Starting tests...
    """)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Testing interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Testing failed: {str(e)}")