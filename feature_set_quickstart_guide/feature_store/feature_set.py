from datetime import datetime

# Import the batch decorator from JFrogML's feature_store module
from frogml.feature_store.feature_sets import batch

# Import the SQL Spark Transformation to be used as output in our feature set method definition
from frogml.core.feature_store.feature_sets.transformations import SparkSqlTransformation


# Constants to use across the project to identify FeatureSet objects
FEATURE_SET = "user-credit-risk-features"
ENTITY_KEY = "user_id"


"""
@batch.feature_set()    -> Defining the FeatureSet with its Data Source and Entity key

@batch.metadata()       -> Additional context into what the Feature Set does and who created it

@batch.scheduling()     -> Schedule the Feature Set job to run daily at midnight 

@batch.backfill()       -> Backfill all the data starting with 1st Jan 2015 until today.
"""
@batch.feature_set(
    name=FEATURE_SET,
    key=ENTITY_KEY,
    data_sources=["credit_risk_data"],
)
@batch.metadata(
    owner="John Doe",
    display_name="User Credit Risk Features",
    description="Features describing user credit risk",
)
@batch.scheduling(cron_expression="0 0 * * *")
@batch.backfill(start_date=datetime(2015, 1, 1))
def user_features():
    return SparkSqlTransformation(
        """
        SELECT user_id,
               age,
               sex,
               job,
               housing,
               saving_account,
               checking_account,
               credit_amount,
               duration,
               purpose,
               date_created
        FROM credit_risk_data
        """
    )