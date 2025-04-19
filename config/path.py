### path ###
from sitecustomize import ROOT
paths = {
    "train": ROOT + "/data/pkl/application_train.p",
    "test": ROOT + "/data/pkl/application_test.p",
    "prev": ROOT + "/data/pkl/previous_application.p",
    "installments": ROOT + "/data/pkl/installments_payments.p",
    "credit_card": ROOT + "/data/pkl/credit_card_balance.p",
    "pos_cash": ROOT + "/data/pkl/pos_cash_balance.p",
    "bureau": ROOT + "/data/pkl/bureau.p",
    "bureau_balance": ROOT + "/data/pkl/bureau_balance.p",
    "description": ROOT + "/data/pkl/description.p"
}

processed_paths = {
    "train": ROOT + "/data/processed/train",
    "test": ROOT + "/data/processed/test",
    "prev": ROOT + "/data/processed/prev",
    "installments": ROOT + "/data/processed/installments",
    "credit_card": ROOT + "/data/processed/credit_card",
    "pos_cash": ROOT + "/data/processed/pos_cash",
    "bureau": ROOT + "/data/processed/bureau",
    "bureau_balance": ROOT + "/data/processed/bureau_balance",
    "description": ROOT + "/data/processed/description"
}

feature_paths = {
    "train": ROOT + "/data/feature/train",
    "test": ROOT + "/data/feature/test",
    "prev": ROOT + "/data/feature/prev",
    "installments": ROOT + "/data/feature/installments",
    "credit_card": ROOT + "/data/feature/credit_card",
    "pos_cash": ROOT + "/data/feature/pos_cash",
    "bureau": ROOT + "/data/feature/bureau",
    "bureau_balance": ROOT + "/data/feature/bureau_balance",
    "description": ROOT + "/data/feature/description"
}