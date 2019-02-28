import numpy as np
import pandas as pd

from settings.basic import DATE_FORMAT

IB_fees_x_share = 0.005

min_fee = 1  # 1 euro
max_fee_percent = 0.01  # 1% of the trade value


class Portfolio(object):
    def __init__(self, day, cash, positions):
        self.positions = []
        self._day_str = pd.to_datetime(str(day)).strftime(DATE_FORMAT)
        self.cash = cash

        if len(positions) > 0:
            self.positions = [p for p in positions if p is not None]

        self.long = len(
            [p for p in self.positions if p.position == Position.LONG])
        self.short = len(
            [p for p in self.positions if p.position == Position.SHORT])

    @property
    def total_money(self) -> float:
        return self.cash + self.invested_money

    @property
    def net_invested(self) -> float:
        return np.sum([p.net_value for p in self.positions])

    @property
    def fees(self) -> float:
        return np.sum([p.fees for p in self.positions])

    @property
    def invested_money(self) -> float:
        return np.sum([p.value for p in self.positions])

    @property
    def available_money(self) -> float:
        return self.cash

    def __str__(self):
        str_rep = "%s: Positions: L %s - S %s. Available: %.2f, Invested: %.2f [%.2f + %.2f], Total: %.2f\n" % (
            self._day_str, self.long, self.short, self.available_money,
            self.invested_money, self.net_invested, self.fees, self.total_money)

        for p in self.positions:
            str_rep += '\n' + str(p)
        return str_rep

    def __repr__(self):
        return "%s: Positions: L %s - S %s. Available: %.2f, Invested: %.2f [%.2f + %.2f], Total: %.2f\n" % (
            self._day_str, self.long, self.short, self.available_money,
            self.invested_money, self.net_invested, self.fees, self.total_money)


class Position(object):
    LONG = 'L'
    SHORT = 'S'

    def __init__(self, symbol, position, shares, buy_price, current_price):
        self.symbol = symbol
        self.position = position
        self.shares = shares
        self.buy_price = buy_price
        self.current_price = current_price

    def update_price(self, current_price):
        # we keep a position, basically we update its current price and return
        # a new instance
        return Position(self.symbol, self.position, self.shares,
                        self.buy_price, current_price)

    @staticmethod
    def get_fees(num_shares, price):
        """ Get the Interactive brokers applied fees for a tx of num_shares
        at given price. """
        try:
            fee = num_shares * IB_fees_x_share
            if fee < min_fee:
                fee = min_fee
            max_fee = (max_fee_percent * num_shares * price)
            if fee > max_fee:
                fee = max_fee

            return fee
        except TypeError as e:
            raise Exception(
                "%s\nNum shares: %s. Price: %s" % (e, num_shares, price))

    @staticmethod
    def get_num_shares(price, money):
        """ Returns how many shares can be bought with money at given price and
        the extra money resulting of buying the shares AND paying the fees."""

        def buy(num_shares_, price_, money_):
            fees = Position.get_fees(num_shares_, price_)
            return money_ - (num_shares_ * price_ + fees)

        first = 0
        last = money / price
        num_shares = 0
        extra = money
        while first <= last:
            candidate_num = (first + last) // 2
            new_extra = buy(candidate_num, price, money)
            if new_extra > 0:
                first = candidate_num + 1
                num_shares = candidate_num
                extra = new_extra
            else:
                last = candidate_num - 1

        return int(num_shares), extra

    @staticmethod
    def long(symbol, price, available_money):
        return Position._open_position(symbol, price, available_money,
                                       Position.LONG)

    @staticmethod
    def short(symbol, price, available_money):
        return Position._open_position(symbol, price, available_money,
                                       Position.SHORT)

    @staticmethod
    def _open_position(symbol, price, available_money, position):
        num_shares, extra = Position.get_num_shares(price, available_money)
        if num_shares == 0:
            print("Can't buy shares of %s [available, price]: [%s, %s]" % (
                symbol, available_money, price))
            return None, extra

        return Position(symbol, position, num_shares, buy_price=price,
                        current_price=price), extra

    def sell(self, current_price):
        """ Returns the total money (initial investment + benefits) of selling
        the position at current_price having applied the fees. """

        fees = self.get_fees(self.shares, current_price)

        initial_price = self.shares * self.buy_price
        end_price = self.shares * current_price

        if self.is_long():
            profit = end_price - initial_price

        else:  # position is short
            profit = initial_price - end_price

        total = initial_price + profit - fees

        return total

    @property
    def fees(self):
        return self.get_fees(self.shares, self.current_price)

    @property
    def net_value(self):
        return self.sell(self.current_price)

    @property
    def value(self):
        if self.is_long():
            return self.shares * (
                self.buy_price + (self.current_price - self.buy_price))
        else:
            return self.shares * (
                self.buy_price + (self.buy_price - self.current_price))

    def is_long(self):
        return self.position == Position.LONG

    def is_short(self):
        return self.position == Position.SHORT

    def __str__(self):
        return "\n[%s, %s] %s shares. Buy/Current: %.1f/%.1f. Total money: %.1f [%.1f, + %.1f]" % (
            self.symbol, self.position, self.shares, self.buy_price,
            self.current_price, self.value, self.net_value, self.fees)

    def __repr__(self):
        return "%s (%s.%s)" % (self.symbol, self.shares, self.position)
