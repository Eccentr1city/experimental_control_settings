#!/usr/bin/env python3
"""
Comprehensive test runner for the compute platform
"""
import sys
import os
import subprocess
from datetime import datetime

def run_test_file(test_file, description):
    """Run a test file and return results"""
    print(f"\n{'='*60}")
    print(f"Running {description}")
    print(f"File: {test_file}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([sys.executable, test_file], 
                              capture_output=True, text=True, cwd=os.path.dirname(__file__))
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        success = result.returncode == 0
        print(f"\n{'✅ PASSED' if success else '❌ FAILED'}: {description}")
        return success
        
    except Exception as e:
        print(f"❌ ERROR running {test_file}: {e}")
        return False


def main():
    """Run all tests"""
    print("🚀 Compute Platform - Comprehensive Test Suite")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Define test files and descriptions
    tests = [
        ("test_models.py", "Data Models Validation"),
        ("test_simple_api.py", "Basic API Functionality"),
        ("test_comprehensive.py", "Comprehensive Model Testing"),
        ("test_integration.py", "Integration & Real-World Scenarios")
    ]
    
    results = []
    total_tests = len(tests)
    
    for test_file, description in tests:
        if os.path.exists(test_file):
            success = run_test_file(test_file, description)
            results.append((description, success))
        else:
            print(f"⚠️ Test file not found: {test_file}")
            results.append((description, False))
    
    # Summary
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, success in results if success)
    
    for description, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status}: {description}")
    
    print(f"\n📈 Overall Results: {passed}/{total_tests} test suites passed")
    
    if passed == total_tests:
        print("🎉 ALL TESTS PASSED! The compute platform is working correctly.")
        print("\n🔧 Platform Status:")
        print("   ✅ Data models are valid and functional")
        print("   ✅ Core services can be instantiated")
        print("   ✅ JSON serialization works properly")
        print("   ✅ Model validation is working")
        print("   ✅ Complex workflows can be created")
        print("   ✅ Multi-user scenarios work")
        print("   ✅ Resource management is functional")
        print("   ✅ Pipeline orchestration works")
        
        print("\n🚀 Ready for:")
        print("   • ML experiment tracking")
        print("   • Model deployment and serving")
        print("   • Data pipeline orchestration")
        print("   • Multi-user collaboration")
        print("   • Resource allocation and management")
        
        print("\n⚠️ Note: This is a testing environment.")
        print("   For production use, implement:")
        print("   • Database persistence")
        print("   • Authentication/authorization")
        print("   • API security")
        print("   • Monitoring and logging")
        print("   • Error handling and recovery")
        
    else:
        print(f"⚠️ {total_tests - passed} test suite(s) failed. Review the output above.")
        print("\n🔍 Common issues to check:")
        print("   • Missing dependencies")
        print("   • Import path issues")
        print("   • Model validation errors")
        print("   • Service configuration problems")
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return passed == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 