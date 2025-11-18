import sys, json
import kfp
from kfp import dsl

HOST = "http://localhost:8080"

@dsl.component(
    base_image="python:3.9",
    packages_to_install=["pandas==1.2.4","numpy==1.21.0","scikit-learn==0.24.2"]
)
def predict_rental_price_model() -> float:
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    df = pd.read_csv("https://raw.githubusercontent.com/mrsddq/MLOps-Project/refs/heads/master/src/data/housing_1000.csv")
    X = df[["rooms","sqft"]].values
    y = df["price"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    return float(LinearRegression().fit(X_train, y_train).predict(X_test[:1])[0])

@dsl.pipeline(name="rental-price-prediction-pipeline")
def pipeline() -> float:
    return predict_rental_price_model().output

def ping(url: str) -> bool:
    try:
        import urllib.request
        with urllib.request.urlopen(url, timeout=3) as r:
            r.read(64)
        return True
    except Exception:
        return False

def main():
    print(f"Python: {sys.version.split()[0]}")
    print(f"kfp version: {getattr(kfp, '__version__', 'unknown')}")

    # 1) Server reachability
    if not ping(HOST):
        print(f"\nERROR: Cannot reach {HOST}. Is your KFP UI/API reachable or port-forward running?\n"
              f"Example: kubectl port-forward -n kubeflow svc/ml-pipeline 8080:8888")
        sys.exit(1)

    # 2) Detect API flavor (v2 vs v1)
    v2_ok = ping(f"{HOST}/apis/v2beta1/healthz")
    v1_ok = ping(f"{HOST}/apis/v1beta1/healthz")
    print(f"Server v2 endpoint: {'OK' if v2_ok else 'NO'}  |  v1 endpoint: {'OK' if v1_ok else 'NO'}")

    try:
        if v2_ok:
            # --- KFP v2 path ---
            from kfp import compiler, client as kfp_client
            pkg = "rental_price_prediction_pipeline.json"
            compiler.Compiler().compile(pipeline, package_path=pkg)
            c = kfp_client.Client(host=HOST)
            run = c.create_run_from_pipeline_package(pkg, run_name="rental-price-run", arguments={})
            print("Run submitted (v2):", run)
        elif v1_ok:
            # --- KFP v1 path ---
            from kfp import compiler
            pkg = "rental_price_prediction_pipeline.yaml"
            compiler.Compiler().compile(pipeline, package_path=pkg)
            c = kfp.Client(host=HOST)
            exp = c.create_experiment(name="Predict Rental Price Experiment")
            run = c.create_run_from_pipeline_func(
                pipeline_func=pipeline,
                experiment_name=exp.display_name,
                run_name="Run of Rental Price Prediction Pipeline",
                arguments={}
            )
            print("Run submitted (v1):", run)
        else:
            print("ERROR: Server reachable, but neither v1 nor v2 health endpoint responded. "
                  "Check the base path or auth/proxy.")
            sys.exit(1)

    except Exception as e:
        # Show the real cause instead of a generic exit code
        print("\n=== TRACE ===")
        import traceback; traceback.print_exc()
        print("=== /TRACE ===\n")
        print("Hint:\n"
              "- If trace says: 'This client only works with Kubeflow Pipeline v2...' → install kfp==1.8.22 and use v1 path.\n"
              "- If it's ConnectionError → port-forward or HOST is wrong.\n"
              "- If ModuleNotFoundError: kfp → select the right interpreter/venv in VS Code.")
        sys.exit(1)

if __name__ == "__main__":
    main()
