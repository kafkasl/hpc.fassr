from settings.basic import GRAHAM_CRITERIA
import argparse


def get_manager(origin):
    if origin == 'intrinio':
        import data_managers.intrinio as manager
        return manager

    else:
        raise Exception("No data manager specified to obtain stock data.")


def screen(manager, tickers: list, criteria=GRAHAM_CRITERIA):
    if criteria == GRAHAM_CRITERIA:
        from criteria.graham import screen_stocks

        return screen_stocks(manager, tickers)

    raise Exception("Screening criteria not found, nothing was done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--criteria', default="graham",
                        help="Criteria to be used for the stock screening [graham]")

    args = parser.parse_args()

    # tickers = ['GOOGL', 'AAPL', 'MSFT']

    tickers = open('../data/dow30_symbols.lst').read().split()
    tickers = open('../data/sp500_symbols.lst').read().split()

    manager = get_manager('intrinio')
    screened_stocks = screen(manager, tickers=tickers, criteria=args.criteria)

    # print("Stocks that passed the screening %s" % len(screened_stocks))
    # [print(s.id) for s in screened_stocks]

    indicators = screened_stocks[0].screening['graham'].keys()
    print("Indicators:\n%s" % indicators)
    for s in screened_stocks:
        print("%s" % s.id, end='')
        for i in indicators:
            v = s.screening['graham'][i]
            if len(str(v)) > 4:
                print(" %.2f" % v, end='')
            else:
                print(" %s" % v, end='')
        print()



