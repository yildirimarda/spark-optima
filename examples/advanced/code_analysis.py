#!/usr/bin/env python3
"""Advanced example: Analyzing and improving Spark code."""

from spark_optima.analysis import CodeAnalyzer


def main():
    """Analyze Spark code and get improvement suggestions."""
    print("=" * 70)
    print("🔍 Spark Optima - Code Analysis Example")
    print("=" * 70)

    # Create code analyzer
    analyzer = CodeAnalyzer()

    # Analyze a Spark code file
    print("\nAnalyzing Spark code...")
    analysis = analyzer.analyze_file("./my_spark_job.py")

    # Display code smells
    if analysis.issues:
        print(f"\n⚠️  Found {len(analysis.issues)} potential issues:")
        print("-" * 70)
        for i, issue in enumerate(analysis.issues, 1):
            print(f"{i}. Line {issue.line_number}: {issue.issue_type}")
            print(f"   Severity: {issue.severity}")
            print(f"   {issue.description}")
            print(f"   💡 {issue.suggestion}\n")
    else:
        print("\n✅ No issues found in the code!")

    # Display detected operations
    if analysis.operations:
        print("\n📋 Detected Operations:")
        print("-" * 70)
        for op in analysis.operations:
            print(f"  - {op.operation_type}: {op.description}")

    # Get optimization suggestions
    print("\n💡 Code Optimization Suggestions:")
    print("-" * 70)
    for suggestion in analysis.suggestions[:5]:
        print(f"  • {suggestion}")


if __name__ == "__main__":
    main()
