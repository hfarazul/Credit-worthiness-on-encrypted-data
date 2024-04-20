"""Data pre-processing functions."""

import numpy
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler


def _get_pipeline_replace_one_hot(func, value):
    return Pipeline([
        ("replace", FunctionTransformer(
            func,
            kw_args={"value": value},
            feature_names_out='one-to-one',
        )),
        ("one_hot", OneHotEncoder(),),
    ])

def _replace_values_eq(column, value):
    for desired_value, values_to_replace in value.items():
        column = numpy.where(numpy.isin(column, values_to_replace), desired_value, column)
    return column

def get_pre_processors():
    pre_processor_applicant = ColumnTransformer(
        transformers=[
            ('one_hot_others', OneHotEncoder(), ['education_level', 'employment_status']),
            ('standard_scaler', StandardScaler(), ['age', 'dependents', 'total_employment_years', 'annual_salary', 'monthly_salary']),
        ],
        remainder='passthrough',
        verbose_feature_names_out=False,
    )

    pre_processor_bank = ColumnTransformer(
        transformers=[
            ('standard_scaler', StandardScaler(), ['last_6_months_avg_balance', 'monthly_expenses']),
        ],
        remainder='passthrough',
        verbose_feature_names_out=False,
    )

    pre_processor_credit_bureau = ColumnTransformer(
        transformers=[
            ('standard_scaler', StandardScaler(), ['existing_loans_count', 'assets_value', 'CIBIL_score', 'total_credit_limit', 'current_total_balances', 'credit_utilization_ratio', 'payment_delays', 'recent_enquiries']),
            ('one_hot_encoder', OneHotEncoder(), ['payment_history']),
        ],
        remainder='passthrough',
        verbose_feature_names_out=False,
    )

    return pre_processor_applicant, pre_processor_bank, pre_processor_credit_bureau
