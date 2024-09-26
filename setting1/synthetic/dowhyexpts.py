from dowhy import CausalModel
import pandas as pd
import random
from sklearn.preprocessing import LabelEncoder
from generate_data import generate_observational_data
from config import Config

encoder = LabelEncoder()
cfg=Config()

for n in range(len(cfg.N)):
    D_obs = generate_observational_data(cfg.N[n])
    df = pd.DataFrame(D_obs, columns=['z','x','y'])
    for col in df.columns:
        df[col] = encoder.fit_transform(df[col])
    model = CausalModel(
        data=df,
        treatment="x",
        outcome="y",
        common_causes = []
    )
    # Step 2: Identify causal effect and return target estimands
    identified_estimand = model.identify_effect()

    estimate = model.estimate_effect(identified_estimand,
                                    method_name="backdoor.linear_regression",
                                    )
    print(estimate)
