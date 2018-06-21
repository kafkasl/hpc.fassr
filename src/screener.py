from settings.basic import GRAHAM_CRITERIA
import argparse




def get_manager(origin):

    if origin == "fundamental":
        import data_managers.sqlite as manager
        return manager

    else:
        raise Exception("No data manager specified to obtain stock data.")


def screen(manager, criteria=GRAHAM_CRITERIA):

    if criteria == GRAHAM_CRITERIA:
        from criteria.graham import screen_stocks

        return screen_stocks(manager)

    raise Exception("Screening criteria not found, nothing was done.")




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_origin', default="fundamental",
                        help="Origin of the data to be used. [fundamental]")
    parser.add_argument('-c', '--criteria', default="graham",
                        help="Criteria to be used for the stock screening [graham]")

    args = parser.parse_args()

    manager = get_manager(args.data_origin)
    screened_stocks = screen(manager, args.criteria)

    print("Stocks that passed the screening %s" % len(screened_stocks))
    [print(s) for s in screened_stocks]
