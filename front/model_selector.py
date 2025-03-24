import datetime
from marko_slsqp.markowitz_slsqp import slsqp

def select_model(
    model: str,
    tickers: list,
    start_date: datetime,
    end_date: datetime,
    bounds : list,
):

    match model:
        case "SLSQP":
            return slsqp(tickers, start_date, end_date, bounds)
        # TODO
        case "GA":
            return slsqp(tickers, start_date, end_date, bounds)
        case "CP_SAT":
            return slsqp(tickers, start_date, end_date, bounds)
        case "MINLP":
            return slsqp(tickers, start_date, end_date, bounds)
        case _:
            print(f"ERROR - model {model} should not exists")
            return []
    return []
