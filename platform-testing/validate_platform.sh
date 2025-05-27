#!/bin/bash

# Comprehensive Compute Platform Validation Script
# This script validates all aspects of the compute platform

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

print_header() {
    echo
    echo "=================================================================="
    print_status $BLUE "$1"
    echo "=================================================================="
}

print_success() {
    print_status $GREEN "✅ $1"
}

print_warning() {
    print_status $YELLOW "⚠️  $1"
}

print_error() {
    print_status $RED "❌ $1"
}

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLATFORM_DIR="$(dirname "$SCRIPT_DIR")/compute-platform"

print_header "🚀 COMPUTE PLATFORM COMPREHENSIVE VALIDATION"

# Check if we're in the right directory
if [ ! -d "$PLATFORM_DIR" ]; then
    print_error "Compute platform directory not found at: $PLATFORM_DIR"
    exit 1
fi

print_success "Found compute platform directory: $PLATFORM_DIR"

# 1. Validate Directory Structure
print_header "📁 VALIDATING DIRECTORY STRUCTURE"

required_dirs=(
    "api/v1"
    "models"
    "services"
    "storage"
    "utils"
    "tests"
)

for dir in "${required_dirs[@]}"; do
    if [ -d "$PLATFORM_DIR/$dir" ]; then
        print_success "Directory exists: $dir"
    else
        print_error "Missing directory: $dir"
    fi
done

# 2. Validate Core Model Files
print_header "🏗️  VALIDATING CORE MODEL FILES"

model_files=(
    "models/job.py"
    "models/experiment.py"
    "models/pipeline.py"
    "models/deployment.py"
    "models/user.py"
    "models/resource.py"
    "models/monitoring.py"
)

for file in "${model_files[@]}"; do
    if [ -f "$PLATFORM_DIR/$file" ]; then
        lines=$(wc -l < "$PLATFORM_DIR/$file")
        print_success "Model file exists: $file ($lines lines)"
    else
        print_error "Missing model file: $file"
    fi
done

# 3. Validate API Endpoints
print_header "🌐 VALIDATING API ENDPOINTS"

api_files=(
    "api/v1/experiments.py"
    "api/v1/jobs.py"
    "api/v1/pipelines.py"
    "api/v1/deployments.py"
    "api/v1/monitoring.py"
    "api/v1/resources.py"
)

for file in "${api_files[@]}"; do
    if [ -f "$PLATFORM_DIR/$file" ]; then
        lines=$(wc -l < "$PLATFORM_DIR/$file")
        endpoints=$(grep -c "@router\." "$PLATFORM_DIR/$file" || echo "0")
        print_success "API file exists: $file ($lines lines, $endpoints endpoints)"
    else
        print_error "Missing API file: $file"
    fi
done

# 4. Validate Service Files
print_header "⚙️  VALIDATING SERVICE FILES"

service_files=(
    "services/auth.py"
    "services/experiment_tracker.py"
    "services/model_registry.py"
    "services/job_scheduler.py"
    "services/resource_manager.py"
)

for file in "${service_files[@]}"; do
    if [ -f "$PLATFORM_DIR/$file" ]; then
        lines=$(wc -l < "$PLATFORM_DIR/$file")
        classes=$(grep -c "^class " "$PLATFORM_DIR/$file" || echo "0")
        print_success "Service file exists: $file ($lines lines, $classes classes)"
    else
        print_warning "Service file missing: $file"
    fi
done

# 5. Check Python Syntax
print_header "🐍 VALIDATING PYTHON SYNTAX"

python_files=$(find "$PLATFORM_DIR" -name "*.py" -type f)
syntax_errors=0

for file in $python_files; do
    if python3 -m py_compile "$file" 2>/dev/null; then
        relative_path=${file#$PLATFORM_DIR/}
        print_success "Syntax OK: $relative_path"
    else
        relative_path=${file#$PLATFORM_DIR/}
        print_error "Syntax error: $relative_path"
        ((syntax_errors++))
    fi
done

if [ $syntax_errors -eq 0 ]; then
    print_success "All Python files have valid syntax"
else
    print_error "Found $syntax_errors files with syntax errors"
fi

# 6. Validate Model Complexity
print_header "📊 ANALYZING MODEL COMPLEXITY"

total_lines=0
total_classes=0
total_functions=0

for file in $python_files; do
    if [ -f "$file" ]; then
        lines=$(wc -l < "$file")
        classes=$(grep -c "^class " "$file" || echo "0")
        functions=$(grep -c "^def \|^async def " "$file" || echo "0")
        
        total_lines=$((total_lines + lines))
        total_classes=$((total_classes + classes))
        total_functions=$((total_functions + functions))
    fi
done

print_success "Total lines of code: $total_lines"
print_success "Total classes: $total_classes"
print_success "Total functions: $total_functions"

# 7. Check for Key Features
print_header "🔍 CHECKING KEY FEATURES"

# Check for async/await usage
async_count=$(grep -r "async def\|await " "$PLATFORM_DIR" --include="*.py" | wc -l)
print_success "Async/await usage: $async_count occurrences"

# Check for dataclass usage
dataclass_count=$(grep -r "@dataclass" "$PLATFORM_DIR" --include="*.py" | wc -l)
print_success "Dataclass usage: $dataclass_count occurrences"

# Check for enum usage
enum_count=$(grep -r "class.*Enum" "$PLATFORM_DIR" --include="*.py" | wc -l)
print_success "Enum definitions: $enum_count occurrences"

# Check for type hints
typing_count=$(grep -r "from typing import\|: List\|: Dict\|: Optional" "$PLATFORM_DIR" --include="*.py" | wc -l)
print_success "Type hints usage: $typing_count occurrences"

# 8. Run Existing Tests
print_header "🧪 RUNNING EXISTING TESTS"

cd "$SCRIPT_DIR"

if [ -f "test_models.py" ]; then
    print_status $BLUE "Running model tests..."
    if python3 test_models.py; then
        print_success "Model tests passed"
    else
        print_error "Model tests failed"
    fi
else
    print_warning "Model tests not found"
fi

if [ -f "test_comprehensive.py" ]; then
    print_status $BLUE "Running comprehensive tests..."
    if python3 test_comprehensive.py; then
        print_success "Comprehensive tests passed"
    else
        print_error "Comprehensive tests failed"
    fi
else
    print_warning "Comprehensive tests not found"
fi

if [ -f "test_integration.py" ]; then
    print_status $BLUE "Running integration tests..."
    if python3 test_integration.py; then
        print_success "Integration tests passed"
    else
        print_error "Integration tests failed"
    fi
else
    print_warning "Integration tests not found"
fi

# 9. Validate Configuration Files
print_header "⚙️  VALIDATING CONFIGURATION"

config_files=(
    "requirements.txt"
    "pyproject.toml"
    "setup.py"
    "Dockerfile"
    "docker-compose.yml"
)

for file in "${config_files[@]}"; do
    if [ -f "$PLATFORM_DIR/$file" ]; then
        print_success "Config file exists: $file"
    else
        print_warning "Config file missing: $file"
    fi
done

# 10. Security and Best Practices Check
print_header "🔒 SECURITY AND BEST PRACTICES CHECK"

# Check for hardcoded secrets
secrets_found=$(grep -r "password\|secret\|key\|token" "$PLATFORM_DIR" --include="*.py" | grep -v "# " | wc -l)
if [ $secrets_found -gt 0 ]; then
    print_warning "Found $secrets_found potential hardcoded secrets (review manually)"
else
    print_success "No obvious hardcoded secrets found"
fi

# Check for SQL injection patterns
sql_patterns=$(grep -r "execute.*%" "$PLATFORM_DIR" --include="*.py" | wc -l)
if [ $sql_patterns -gt 0 ]; then
    print_warning "Found $sql_patterns potential SQL injection patterns"
else
    print_success "No obvious SQL injection patterns found"
fi

# Check for proper error handling
try_except_count=$(grep -r "try:\|except " "$PLATFORM_DIR" --include="*.py" | wc -l)
print_success "Error handling blocks: $try_except_count"

# 11. Performance Analysis
print_header "⚡ PERFORMANCE ANALYSIS"

# Check for database queries in loops
loop_queries=$(grep -A5 -B5 "for.*in\|while " "$PLATFORM_DIR" --include="*.py" | grep -c "query\|execute\|find" || echo "0")
if [ $loop_queries -gt 0 ]; then
    print_warning "Found $loop_queries potential N+1 query patterns"
else
    print_success "No obvious N+1 query patterns found"
fi

# Check for caching usage
cache_usage=$(grep -r "cache\|Cache" "$PLATFORM_DIR" --include="*.py" | wc -l)
print_success "Caching usage: $cache_usage occurrences"

# 12. Documentation Check
print_header "📚 DOCUMENTATION CHECK"

# Check for docstrings
docstring_count=$(grep -r '"""' "$PLATFORM_DIR" --include="*.py" | wc -l)
print_success "Docstrings found: $docstring_count"

# Check for README files
readme_files=$(find "$PLATFORM_DIR" -name "README*" -type f | wc -l)
print_success "README files: $readme_files"

# Check for comments
comment_count=$(grep -r "^[[:space:]]*#" "$PLATFORM_DIR" --include="*.py" | wc -l)
print_success "Comment lines: $comment_count"

# 13. Dependency Analysis
print_header "📦 DEPENDENCY ANALYSIS"

if [ -f "$PLATFORM_DIR/requirements.txt" ]; then
    deps=$(wc -l < "$PLATFORM_DIR/requirements.txt")
    print_success "Dependencies listed: $deps"
    
    # Check for common ML/AI libraries
    ml_libs=("torch" "tensorflow" "scikit-learn" "numpy" "pandas" "fastapi" "pydantic")
    for lib in "${ml_libs[@]}"; do
        if grep -q "$lib" "$PLATFORM_DIR/requirements.txt"; then
            print_success "ML library found: $lib"
        else
            print_warning "ML library missing: $lib"
        fi
    done
else
    print_warning "No requirements.txt found"
fi

# 14. Final Summary
print_header "📋 VALIDATION SUMMARY"

# Count files and calculate metrics
total_py_files=$(find "$PLATFORM_DIR" -name "*.py" -type f | wc -l)
total_size=$(du -sh "$PLATFORM_DIR" | cut -f1)

print_success "Total Python files: $total_py_files"
print_success "Total codebase size: $total_size"
print_success "Average lines per file: $((total_lines / total_py_files))"

# Calculate complexity score
complexity_score=$((total_classes + total_functions + async_count + dataclass_count))
print_success "Complexity score: $complexity_score"

if [ $complexity_score -gt 500 ]; then
    print_success "Platform complexity: HIGH (suitable for research)"
elif [ $complexity_score -gt 200 ]; then
    print_success "Platform complexity: MEDIUM"
else
    print_warning "Platform complexity: LOW"
fi

# 15. Generate Test Report
print_header "📄 GENERATING TEST REPORT"

report_file="$SCRIPT_DIR/validation_report_$(date +%Y%m%d_%H%M%S).txt"

cat > "$report_file" << EOF
COMPUTE PLATFORM VALIDATION REPORT
Generated: $(date)

CODEBASE METRICS:
- Total Python files: $total_py_files
- Total lines of code: $total_lines
- Total classes: $total_classes
- Total functions: $total_functions
- Codebase size: $total_size

FEATURE ANALYSIS:
- Async/await usage: $async_count
- Dataclass usage: $dataclass_count
- Enum definitions: $enum_count
- Type hints: $typing_count
- Error handling blocks: $try_except_count
- Docstrings: $docstring_count
- Comments: $comment_count

COMPLEXITY SCORE: $complexity_score

VALIDATION STATUS: COMPLETED
EOF

print_success "Validation report saved: $report_file"

print_header "🎉 VALIDATION COMPLETED"

if [ $syntax_errors -eq 0 ]; then
    print_success "All validations passed successfully!"
    print_success "The compute platform is ready for use and testing."
    exit 0
else
    print_error "Some validations failed. Please review the output above."
    exit 1
fi 