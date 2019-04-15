from math import sqrt

import pandas as pd

from models.portfolio import Portfolio, Position


def get_share_price(prices, day, symbol):
    return prices.loc(axis=0)[(symbol, day)]


def debug_selector(df_trade, day, clf_name, trading_params):
    return df_trade.loc[[day]].query('symbol=="EIX"')[
               ['y', 'y', 'symbol', 'price']], \
           df_trade.loc[[day]].query('symbol=="OKE"')[
               ['y', 'y', 'symbol', 'price']]


def get_k_best(df_trade, day, clf_name, trading_params):
    # sort by predicted increment per stock of classifier clf_name
    df_clf = df_trade[['y', clf_name, 'symbol', 'price']]
    df_aux = df_clf.loc[[day]].sort_values(clf_name)

    k = trading_params['k']
    top_thresh = trading_params['top_thresh']
    bot_thresh = trading_params['bot_thresh']

    botk = df_aux.iloc[0:k].query('%s<%s' % (clf_name, bot_thresh))
    topk = df_aux.iloc[-k - 1: -1].query('%s>%s' % (clf_name, top_thresh))

    return topk, botk


def get_k_random(df_trade, day, clf_name, trading_params):
    df_trade = df_trade[['y', 'y', 'symbol', 'price']]
    k = trading_params['k']

    sample_size = min(len(df_trade), 2 * k)
    samples = df_trade.sample(n=sample_size)
    topk = samples.iloc[:len(samples) // 2]
    botk = samples.iloc[len(samples) // 2:]
    return topk, botk


def graham_screening(df_trade, day, clf_name, trading_params):
    df_y = (df_trade
            .groupby('symbol')
            .resample('1Y')
            .ffill()
            .drop('symbol', axis=1)
            .reset_index()
            .set_index('date')
            .sort_index())

    symbols = list(set(df_trade.symbol))
    screened_symbols = []
    empty_df = pd.DataFrame(columns=df_trade.columns)

    for symbol in symbols:
        df_aux = df_trade.loc[:day + 1].query('symbol=="%s"' % symbol)
        # we have df with only yearly data to compute the last 10/20 years
        # conditions
        df_aux_y = df_y.loc[:day + 1].query('symbol=="%s"' % symbol)

        row_years = df_aux_y.shape[0]

        if row_years == 0:
            continue

        succesful = True

        try:
            day_info = df_aux.loc[day]
        except KeyError:
            # This symbol has no fundamental data for the current day. We just
            # skip it.
            continue

        # TODO where is the 2 * assets > liabilities

        succesful &= day_info.currentratio > 2
        succesful &= day_info.revenue > 1500000000
        succesful &= day_info.wc > 0
        succesful &= (df_aux_y.eps > 0).sum() == row_years
        succesful &= (df_aux_y.divpayoutratio > 0).sum() == row_years
        succesful &= day_info.epsgrowth > 0.03

        try:
            pt = sqrt(22.5 * df_aux.loc[day].p2e * df_aux.loc[day].bvps)
            succesful &= df_aux.loc[day].price < pt
        except ValueError as e:
            # A negative graham value is not a problem, just don't square it
            # print("Negative graham value: %s" % (
            #     22.5 * df_aux.loc[day].p2e * df_aux.loc[day].bvps))
            continue

        if succesful:
            screened_symbols.append(symbol)

    if len(screened_symbols) > 0:
        chosen = df_trade.loc[day].query('symbol in @screened_symbols')
        topk, botk = chosen[['y', 'y', 'symbol', 'price']], empty_df
    else:
        topk, botk = empty_df, empty_df

    return topk, botk


def update_positions_buy_sell_all(prices, day, available_money, positions,
                                  botk, topk):
    new_positions = []
    print("Start %s with %s =================================" % (
        day, available_money))
    print("Old positions: %s" % positions)
    print("Recommended top %s: %s" % (len(topk), list(topk.symbol)))
    print("Recommended bot %s: %s" % (len(botk), list(botk.symbol)))
    # Sell all previous positions
    same_positions = len(positions) > 0
    for p in positions:
        if p.is_long():
            same_positions &= p.symbol in list(topk.symbol)
        elif p.is_short():
            same_positions &= p.symbol in list(botk.symbol)
        if not same_positions:
            break
    # The new positions are equal to the old ones, avoid fees by returning same
    # positions and available money.
    if same_positions:
        print("Same positions today as previous day." % day)
        for p in positions:
            try:
                current_price = get_share_price(prices, day, p.symbol)
            except Exception as e:
                # This happens when we try to trade on a weekend or similar,
                # we just use last close price.
                # print("New price for %s not available using last." % p.symbol)
                current_price = p.current_price

            # get share price to update the position current_price, other
            # than that the position is exactly the same
            new_position = p.update_price(current_price)

            # we do not pay fees as we already had that stock and position
            new_positions.append(new_position)

    else:
        for p in positions:
            try:
                current_price = get_share_price(prices, day, p.symbol)
            except Exception as e:
                # This happens when we try to trade on a weekend or similar,
                # we just use last close price.
                print("New price for %s not available using last." % p.symbol)
                current_price = p.current_price

            # we sell the positions that we do not continue
            # import ipdb
            # ipdb.set_trace()
            available_money += p.sell(current_price)

        # Buy all new positions
        if botk.shape[0] > 0 or topk.shape[0] > 0:
            # initially divide money equally among candidates
            remaining_stocks = (botk.shape[0] + topk.shape[0])

            for (idx, y, pred, symbol, price) in topk.itertuples():
                # get a proportional amount of money to buy
                stash = available_money / remaining_stocks
                # subtract it from the total
                available_money -= stash

                position, extra_money = Position.long(symbol=symbol,
                                                      price=price,
                                                      available_money=stash)
                if position is None:
                    continue
                new_positions.append(position)

                # extra money of buying only whole shares is returned to the
                # stash
                available_money += extra_money
                remaining_stocks -= 1

            for (idx, y, pred, symbol, price) in botk.itertuples():
                # get a proportional amount of money to buy
                stash = available_money / remaining_stocks
                # subtract it from the total
                available_money -= stash

                position, extra_money = Position.short(symbol=symbol,
                                                       price=price,
                                                       available_money=stash)
                if position is None:
                    continue
                new_positions.append(position)

                # extra money of buying only whole shares is returned to the
                # stash
                available_money += extra_money
                remaining_stocks -= 1

    print("New positions: %s" % '\n'.join([str(np) for np in new_positions]))
    print("End %s with %s =================================\n\n" % (
        day, available_money))
    return available_money, new_positions


def update_positions_avoiding_fees(prices, day, available_money, positions,
                                   botk, topk):
    # make sure that 'random' selecting is always the same
    topk = topk.sort_values('symbol')
    botk = botk.sort_values('symbol')
    new_positions = []
    print("Start %s with %s =================================" % (
        day, available_money))
    print("Old positions: %s" % positions)
    print("Recommended top %s: %s" % (len(topk), list(topk.symbol)))
    print("Recommended bot %s: %s" % (len(botk), list(botk.symbol)))
    for p in positions:
        try:
            current_price = get_share_price(prices, day, p.symbol)
        except Exception as e:
            # This happens when we try to trade on a weekend or similar,
            # we just use last close price.
            # print("New price for %s not available using last." % p.symbol)
            current_price = p.current_price

        if (p.symbol in botk.symbol.values and p.is_short()) or \
                (p.symbol in topk.symbol.values and p.is_long()):

            # get share price to update the position current_price, other
            # than that the position is exactly the same
            new_position = p.update_price(current_price)

            # we do not pay fees as we already had that stock and position
            new_positions.append(new_position)

            # remove symbol as its already been processed
            topk = topk[topk.symbol != p.symbol]
            botk = botk[botk.symbol != p.symbol]
        else:
            # we sell the positions that we do not continue
            available_money += p.sell(current_price)

    if (botk.shape[0] > 0 or topk.shape[0] > 0) and len(new_positions) < 20:
        # we want to have a portfolio of 20, so we divide
        stash = available_money / (20 - len(new_positions))

        for (idx, y, pred, symbol, price) in topk.itertuples():
            # get a proportional amount of money to buy
            # subtract it from the total
            if available_money - stash < 0:
                break

            available_money -= stash

            position, extra_money = Position.long(symbol=symbol, price=price,
                                                  available_money=stash)
            if position is None:
                continue
            new_positions.append(position)

            # extra money of buying only whole shares is returned to the stash
            available_money += extra_money

        for (idx, y, pred, symbol, price) in botk.itertuples():
            # get a proportional amount of money to buy
            # subtract it from the total
            if available_money - stash < 0:
                break

            available_money -= stash

            position, extra_money = Position.short(symbol=symbol, price=price,
                                                   available_money=stash)
            if position is None:
                continue
            new_positions.append(position)

            # extra money of buying only whole shares is returned to the stash
            available_money += extra_money

    print("New positions: %s" % new_positions)
    print("End %s with %s =================================\n\n" % (
        day, available_money))
    return available_money, new_positions


def model_trade(df_trade, indices, prices, clf_name, trading_params: dict,
                dates):
    return paper_trade(df_trade=df_trade, indices=indices, prices=prices,
                       clf_name=clf_name,
                       trading_params=trading_params, selector_f=get_k_best)


def random_trade(df_trade, indices, prices, clf_name, trading_params, dates):
    return paper_trade(df_trade=df_trade, indices=indices, prices=prices,
                       clf_name=clf_name,
                       trading_params=trading_params,
                       selector_f=get_k_random)


def debug_trade(df_trade, indices, prices, clf_name, trading_params, dates):
    return paper_trade(df_trade=df_trade, indices=indices, prices=prices,
                       clf_name=clf_name,
                       trading_params=trading_params,
                       selector_f=debug_selector)


def graham_trade(df_trade, indices, prices, clf_name, trading_params, dates):
    return paper_trade(df_trade=df_trade, indices=indices, prices=prices,
                       clf_name=clf_name,
                       trading_params=trading_params,
                       selector_f=graham_screening)


def paper_trade(df_trade, indices, prices, clf_name, trading_params,
                selector_f):
    start_date, final_date = trading_params['dates']
    print("Starting trading for %s: %s from [%s-%s] every %s" % (
        clf_name, trading_params['mode'], start_date, final_date,
        trading_params['trade_frequency']))

    if trading_params['mode'] == 'sell_all':
        update_positions = update_positions_buy_sell_all
    elif trading_params['mode'] == 'avoid_fees':
        update_positions = update_positions_avoiding_fees

    portfolios = []
    positions = []
    stash = 100000

    for i in range(0, len(indices)):
        day = indices[i]
        # sort by predicted increment per stock of classifier clf_name

        topk, botk = selector_f(df_trade, day, clf_name, trading_params)

        old_positions = positions
        old_stash = stash
        stash, positions = update_positions(prices, day, old_stash,
                                            old_positions, botk, topk)

        pf = Portfolio(day, cash=stash, positions=positions)

        portfolios.append(pf)

    print("Finished trading for %s" % clf_name)
    return portfolios
