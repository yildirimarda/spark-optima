#!/usr/bin/env python3
"""Example: Using curated workload templates as configuration baselines.

Spark Optima ships curated baseline configurations for common workload
archetypes (batch ETL, streaming, ML training, interactive analytics) in
``data/templates/*.yaml``. This example shows how to:

1. List all available templates with TemplateRegistry.
2. Inspect one template's curated parameters and their rationale.
3. Layer your own settings on top of a template with ``apply_to()``
   (your values always win over the template baseline).

CLI equivalent:
    spark-optima templates
    spark-optima templates --show etl-batch
"""

from spark_optima.core.templates import TemplateRegistry


def main() -> None:
    """List templates, inspect one, and merge user settings on top."""
    print("=" * 70)
    print("📋 Spark Optima - Workload Templates Example")
    print("=" * 70)

    registry = TemplateRegistry()

    # 1. List all available templates
    print(f"\nAvailable templates ({len(registry)}):")
    print("-" * 70)
    for template in registry.list_templates():
        print(f"  {template.name:15s} {template.display_name}")
        print(f"  {'':15s} {template.description.splitlines()[0]}")

    # 2. Inspect one template in detail
    template = registry.get_template("etl-batch")
    print(f"\n🔍 Template: {template.display_name}")
    print("-" * 70)

    if template.workload_traits:
        print("Workload traits:")
        for trait in template.workload_traits:
            print(f"  - {trait}")

    print("\nCurated configuration (with rationale):")
    for param_name, parameter in template.config.items():
        print(f"  {param_name:50s} = {parameter.value}")
        if parameter.comment:
            print(f"  {'':50s}   # {parameter.comment}")

    if template.recommended_for:
        print("\nRecommended for:")
        for item in template.recommended_for:
            print(f"  ✅ {item}")
    if template.not_recommended_for:
        print("Not recommended for:")
        for item in template.not_recommended_for:
            print(f"  ❌ {item}")

    # 3. Layer your own settings on top of the template baseline.
    # The template is the base layer; user-provided parameters win.
    user_config = {
        "spark.executor.memory": "12g",  # Override the template value
        "spark.app.name": "nightly-sales-etl",  # Add a new parameter
    }
    merged = template.apply_to(user_config)

    print("\n🔧 Merged configuration (template baseline + your overrides):")
    print("-" * 70)
    for key in sorted(merged):
        marker = "(yours)" if key in user_config else "(template)"
        print(f"  {key:50s} = {str(merged[key]):10s} {marker}")

    # Membership and size checks
    print(f"\n'etl-batch' in registry: {'etl-batch' in registry}")
    print(f"Total templates loaded:  {len(registry)}")


if __name__ == "__main__":
    main()
