import logging
from typing import Dict, List, Optional, Tuple

from afang.database.trades_db.base import Session, engine
from afang.database.trades_db.models import Base, Order, TradePosition

# Generate database schema
Base.metadata.create_all(engine)

logger = logging.getLogger(__name__)


class TradesDatabase:
    """Interface to store, retrieve, and manipulate user demo/live trade
    data."""

    def __init__(self) -> None:
        """Initialize the TradesDatabase class."""

        self.session = Session()

    def create_new_position(self, trade_position: TradePosition) -> None:
        """Add a new trade position to the trades' database.

        :param trade_position: trade position to add to the trades' database.
        :return: None
        """

        self.session.add(trade_position)

    def delete_position(self, position_id: int) -> None:
        """Delete a trade position from the trades' database.

        :param position_id: ID of the position to be deleted.
        :return: None
        """

        trade_position = (
            self.session.query(TradePosition)
            .filter(TradePosition.id == position_id)
            .first()
        )
        if trade_position:
            self.session.delete(trade_position)

    def update_position(self, position_id: int, updated_fields: Dict) -> None:
        """Update a trade position in the trades' database.

        :param position_id: ID of the position to be updated.
        :param updated_fields: dict containing fields to be updated and their new values.
        :return: None
        """

        self.session.query(TradePosition).filter(
            TradePosition.id == position_id
        ).update(updated_fields)

    def fetch_position_by_id(self, position_id: int) -> Optional[TradePosition]:
        """Fetch a trade position by ID from the trades' database.

        :param position_id: ID of the position to be fetched.
        :return: Optional[TradePosition]
        """

        trade_position = (
            self.session.query(TradePosition)
            .filter(TradePosition.id == position_id)
            .first()
        )
        if not trade_position:
            logger.warning(
                "Trade position not found in DB. position id: %s", position_id
            )

        return trade_position

    def fetch_positions(
        self, filters: Tuple = tuple(), limit: int = -1
    ) -> List[TradePosition]:
        """Fetch multiple trade positions from the trades' database.

        :param filters: optional. tuple of logical filters to filter trade positions.
        :param limit: max number of positions to return. defaults to returning all matching positions.
        :return: List[TradePosition]
        """

        trade_positions = (
            self.session.query(TradePosition).filter(*filters).limit(limit).all()
        )
        return trade_positions

    def create_new_order(self, order: Order) -> None:
        """Add a new order to the trades' database.

        :param order: trade order to add to the trades' database.
        :return: None
        """

        self.session.add(order)

    def update_order(self, order_id: int, updated_fields: Dict) -> None:
        """Update an order in the trades' database.

        :param order_id: ID of the order to be updated.
        :param updated_fields: dict containing fields to be updated and their new values.
        :return: None
        """

        self.session.query(Order).filter(Order.id == order_id).update(updated_fields)

    def fetch_order_by_id(self, db_order_id: int) -> Optional[Order]:
        """Fetch an order by ID from the trades' database.

        :param db_order_id: DB ID of the order to be fetched.
        :return: Optional[Order]
        """

        order = (
            self.session.query(TradePosition).filter(Order.id == db_order_id).first()
        )
        if not order:
            logger.warning("Order not found in DB. id: %s", db_order_id)

        return order

    def fetch_order_by_exchange_id(self, order_id: str) -> Optional[Order]:
        """Fetch an order by exchange order ID from the trades' database.

        :param order_id: exchange order ID of the order to be fetched.
        :return: Optional[Order]
        """

        order = (
            self.session.query(TradePosition).filter(Order.order_id == order_id).first()
        )
        if not order:
            logger.warning("Order not found in DB. exchange order id: %s", order_id)

        return order

    def fetch_orders(self, filters: Tuple = tuple(), limit: int = -1) -> List[Order]:
        """Fetch multiple orders from the trades' database.

        :param filters: optional. tuple of logical filters to filter orders.
        :param limit: max number of orders to be returned. defaults to returning all matching orders.
        :return: List[Order]
        """

        orders = self.session.query(Order).filter(*filters).limit(limit).all()
        return orders
