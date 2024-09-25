from peewee import *
from peewee import Expression, Value, OP
from playhouse.sqlite_ext import JSONField

database = SqliteDatabase('../data/models.db', pragmas=(('foreign_keys', 'ON'),))

class BaseModel(Model):
    class Meta:
        database = database
        legacy_table_names = False


class TupleField(TextField):
    def db_value(self, value):
        return str(value)

    def python_value(self, value):
        return eval(value)

    def __eq__(self, rhs):
        rhs = Value(rhs, converter=self.db_value, unpack=False)
        return Expression(self, OP.EQ, rhs)

    __hash__ = TextField.__hash__

    # def _e(op):
    #     def inner(self, rhs):
    #         rhs = Value(rhs, converter=self.db_value, unpack=False)
    #         return Expression(self, op, rhs)
    #     return inner
    #
    # __eq__ = _e(OP.EQ)


class Experiments(BaseModel):
    val_size = IntegerField()
    batch_size = IntegerField()
    num_epoch = IntegerField()
    random_seed = IntegerField()
    optimizer = TextField()
    opt_params = JSONField()
    scheduler = TextField()
    sch_params = JSONField()
    in_shape = TupleField()
    out_shape = TupleField()

    class Meta:
        indexes = (
            (('val_size', 'batch_size', 'num_epoch', 'random_seed', 'optimizer', 'opt_params',
               'scheduler', 'sch_params', 'in_shape', 'out_shape'), True),
        )


class Devices(BaseModel):
    name = TextField()


class Models(BaseModel):
    hash = TextField(unique=True)
    flops = IntegerField()
    macs = IntegerField()
    params = IntegerField()


class Runs(BaseModel):
    exp_id = ForeignKeyField(Experiments, backref='runs')
    model_id = ForeignKeyField(Models, backref='runs')
    device_id = ForeignKeyField(Devices, backref='runs')
    val_acc = FloatField()
    train_acc = FloatField()
    time = FloatField()


class ModelEpochs(BaseModel):
    run_id = ForeignKeyField(Runs, backref='epoch')
    loss = FloatField()
    train_acc = FloatField()
    val_acc = FloatField()
    time = FloatField()
    class Meta:
        primary_key = False
