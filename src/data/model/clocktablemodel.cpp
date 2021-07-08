#include <data/model/clocktablemodel.h>

#include <hardware/core/clock/clockmanager.h>

#include <QComboBox>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QMetaEnum>

ClockTableModel::ClockTableModel(QObject *parent) :
    QAbstractTableModel(parent), SettingsStorage(BC::Key::ClockTable::ctKey)
{
    QMetaEnum ct = QMetaEnum::fromType<RfConfig::ClockType>();
    for(int i=0; i<ct.keyCount(); ++i)
    {
        d_clockTypes.append(static_cast<RfConfig::ClockType>(ct.value(i)));
        d_clockConfigs.insert(static_cast<RfConfig::ClockType>(ct.value(i)),{});
    }

    SettingsStorage s(BC::Key::Clock::clockManager);
    QString arrKey(BC::Key::Clock::hwClocks);
    auto count = s.getArraySize(arrKey);
    QStringList keys;
    for(std::size_t i = 0; i < count; ++i)
    {
        ClockHwInfo hw;
        hw.used = false;
        hw.index = d_hwInfo.size();
        hw.hwKey = s.getArrayValue<QString>(arrKey,i,BC::Key::Clock::clockKey,"");
        hw.output = s.getArrayValue<int>(arrKey,i,BC::Key::Clock::clockOutput,0);
        hw.name = s.getArrayValue<QString>(arrKey,i,BC::Key::Clock::clockName,QString("%1-%2").arg(hw.hwKey).arg(hw.output));
        if(!hw.hwKey.isEmpty())
        {
            d_hwInfo << hw;
            if(!keys.contains(hw.hwKey))
                keys << hw.hwKey;
        }
    }

    using namespace BC::Key::ClockTable;
    count = getArraySize(ctClocks);
    for(std::size_t i=0; i<count; ++i)
    {
        auto type = getArrayValue(ctClocks,i,ctClockType,RfConfig::UpLO);
        auto hwKey = getArrayValue(ctClocks,i,ctHwKey,QString(""));
        if(keys.contains(hwKey))
        {
            auto output = getArrayValue(ctClocks,i,ctOutput,0);
            auto op = getArrayValue(ctClocks,i,ctOp,RfConfig::Multiply);
            auto factor = getArrayValue(ctClocks,i,ctFactor,1.0);
            auto freq = getArrayValue(ctClocks,i,ctFreq,0.0);

            d_clockConfigs[type] = {freq,op,factor,hwKey,output};
        }
    }
}

ClockTableModel::~ClockTableModel()
{
    using namespace BC::Key::ClockTable;
    setArray(ctClocks,{});
    int i=0;
    for(auto it = d_clockConfigs.cbegin(); it != d_clockConfigs.cend(); ++it)
    {
        if(it.value().hwKey.isEmpty())
            continue;

        setArrayValue(ctClocks,i,ctClockType,it.key(),false);
        setArrayValue(ctClocks,i,ctHwKey,it.value().hwKey,false);
        setArrayValue(ctClocks,i,ctOutput,it.value().output,false);
        setArrayValue(ctClocks,i,ctOp,it.value().op,false);
        setArrayValue(ctClocks,i,ctFactor,it.value().factor,false);
        setArrayValue(ctClocks,i,ctFreq,it.value().desiredFreqMHz,false);

        ++i;
    }
}

void ClockTableModel::setConfig(const RfConfig c)
{
    d_rfConfig = c;
    d_clockAssignments.clear();



    if(!d_rfConfig.getClocks().isEmpty())
    {
        for(auto it = d_rfConfig.getClocks().constBegin(); it!=d_rfConfig.getClocks().constEnd(); it++)
        {
            for(auto &hw : d_hwInfo)
            {
                if(!hw.hwKey.isEmpty() && it.value().hwKey == hw.hwKey &&
                        it.value().output == hw.output)
                {
                    d_clockAssignments.insert(it.key(),hw.index);
                }
            }
        }
    }

    if(c.d_commonUpDownLO)
        setCommonLo(c.d_commonUpDownLO);

    emit dataChanged(index(0,0),index(d_clockConfigs.size(),5));
}

RfConfig ClockTableModel::getRfConfig() const
{
    return d_rfConfig;
}

void ClockTableModel::setCommonLo(bool b)
{
    d_rfConfig.d_commonUpDownLO = b;
    if(b)
    {
        d_rfConfig.setClockFreqInfo(RfConfig::DownLO,d_rfConfig.getClocks().value(RfConfig::UpLO));
        if(d_clockAssignments.contains(RfConfig::UpLO))
        {
            d_clockAssignments.insert(RfConfig::DownLO,d_clockAssignments.value(RfConfig::UpLO));
        }
    }
    else
    {
        d_rfConfig.setClockHwInfo(RfConfig::DownLO,QString(""),0);
        d_clockAssignments.remove(RfConfig::DownLO);
    }

    int downRow = d_clockTypes.indexOf(RfConfig::DownLO);
    emit dataChanged(index(downRow,1),index(downRow,5));

}



int ClockTableModel::rowCount(const QModelIndex &parent) const
{
    Q_UNUSED(parent)
    return d_clockConfigs.size();
}

int ClockTableModel::columnCount(const QModelIndex &parent) const
{
    Q_UNUSED(parent)
    return 5;
}

QVariant ClockTableModel::data(const QModelIndex &index, int role) const
{
    if(index.row() > d_clockConfigs.size())
        return QVariant();

    if(role == Qt::TextAlignmentRole)
    {
        switch(index.column())
        {
        case 0:
        case 1:
        case 3:
            return QVariant(Qt::AlignLeft|Qt::AlignVCenter);
        case 2:
            return QVariant(Qt::AlignCenter|Qt::AlignVCenter);
        case 4:
            return QVariant(Qt::AlignRight|Qt::AlignVCenter);
        default:
            return QVariant();
        }
    }

    auto type = d_clockTypes.at(index.row());
    RfConfig::ClockFreq c;
    if(d_rfConfig.getClocks().contains(type))
        c = d_rfConfig.getClocks().value(type);
    else
    {
        c.desiredFreqMHz = 0.0;
        c.factor = 1.0;
        c.hwKey = QString("");
        c.op = RfConfig::Multiply;
        c.output = 0;
    }

    if(role == Qt::DisplayRole)
    {
        switch(index.column())
        {
        case 0:
            return QMetaEnum::fromType<RfConfig::ClockType>().valueToKey(type);
        case 1:
            if(d_clockAssignments.contains(type))
                return d_hwInfo.at(d_clockAssignments.value(type)).name;
            else
                return QString("None");
        case 2:
            if(c.op == RfConfig::Multiply)
                return QString::fromUtf16(u"×");
            else
                return QString::fromUtf16(u"÷");
        case 3:
            if(d_rfConfig.getClocks().value(type).op == RfConfig::Multiply)
                return QString::fromUtf16(u"×").append(QString::number(c.factor,'f',0));
            else
                return QString::fromUtf16(u"÷").append(QString::number(c.factor,'f',0));
        case 4:
            return QString::number(c.desiredFreqMHz,'f',6);
        default:
            return QVariant();
        }
    }

    if(role == Qt::EditRole)
    {
        switch(index.column())
        {
        case 0:
            return QVariant();
        case 1:
            if(d_clockAssignments.contains(type))
                return d_clockAssignments.value(type);
            else
                return -1;
        case 2:
            return c.op;
        case 3:
            return c.factor;
        case 4:
            return c.desiredFreqMHz;
        default:
            return QVariant();
        }
    }

    if(role == Qt::ToolTipRole)
    {
        switch (index.column()) {
        case 1:
            return QString("The hardware clock used for the %1.\nSelect \"None\" if not applicable.")
                    .arg(QMetaEnum::fromType<RfConfig::ClockType>()
                         .valueToKey(d_clockTypes.at(index.row())));
        case 2:
            return QString("Select whether a frequency multiplier or divider is used on this clock.\nIf none, you may select either and enter a factor of 1.");
        case 3:
            return QString("Multiplier/divider factor for this clock. Used to convert between desired frequency and raw clock frequency.\nIf no multiplier or divider, enter 1.");
        case 4:
            return QString("Desired frequency for this clock in MHz. BlackChirp will convert to the required clock frequency using the multiplier/divider settings.");
        default:
            return QVariant();
        }
    }

    return QVariant();

}

bool ClockTableModel::setData(const QModelIndex &index, const QVariant &value, int role)
{
    if(role != Qt::EditRole)
        return false;

    if(index.row() > d_clockConfigs.size())
        return false;

    auto type = d_clockTypes.at(index.row());

    switch(index.column())
    {
    case 1:
        if(d_clockAssignments.contains(type))
        {
            d_hwInfo[d_clockAssignments.value(type)].used = false;
            d_clockAssignments.remove(type);
        }
        if(value.toInt() < 0)
            d_rfConfig.setClockHwInfo(type,QString(""),0);
        else
        {
            d_hwInfo[value.toInt()].used = true;
            d_clockAssignments.insert(type,value.toInt());
            if(type == d_clockTypes.indexOf(RfConfig::UpLO) && d_rfConfig.d_commonUpDownLO)
                d_clockAssignments.insert(RfConfig::DownLO,value.toInt());

            d_rfConfig.setClockHwInfo(type,d_hwInfo.at(value.toInt()).hwKey,d_hwInfo.at(value.toInt()).output);
        }
        break;
    case 2:
        d_rfConfig.setClockOp(type,static_cast<RfConfig::MultOperation>(value.toInt()));
        break;
    case 3:
        d_rfConfig.setClockFactor(type,value.toDouble());
        break;
    case 4:
        d_rfConfig.setClockDesiredFreq(type,value.toDouble());
        break;
    default:
        return false;
    }

    emit dataChanged(index,index);
    return true;
}

QVariant ClockTableModel::headerData(int section, Qt::Orientation orientation, int role) const
{
    if(orientation == Qt::Horizontal)
    {
        if(role == Qt::DisplayRole)
        {
            switch(section)
            {
            case 0:
                return QString("Clock");
            case 1:
                return QString("Hardware");
            case 2:
                return QString("M/D");
            case 3:
                return QString("Factor");
            case 4:
                return QString("Frequency (MHz)");
            }
        }
    }

    return QVariant();
}

Qt::ItemFlags ClockTableModel::flags(const QModelIndex &index) const
{
    if(index.row() < d_clockConfigs.size())
    {
        if(d_rfConfig.d_commonUpDownLO && index.row() == d_clockTypes.indexOf(RfConfig::DownLO) && index.column() > 0)
            return 0;

        if(index.column() > 0)
            return Qt::ItemIsEnabled|Qt::ItemIsEditable;
    }

    return Qt::ItemIsEnabled;
}

ClockTableDelegate::ClockTableDelegate(QObject *parent) : QStyledItemDelegate(parent)
{
}

QWidget *ClockTableDelegate::createEditor(QWidget *parent, const QStyleOptionViewItem &option, const QModelIndex &index) const
{
    Q_UNUSED(option)

    QWidget *out = nullptr;

    if(index.column() == 1)
    {
        auto l = dynamic_cast<const ClockTableModel*>(index.model())->getHwInfo();
        QComboBox *cb = new QComboBox(parent);
        cb->addItem(QString("None"),QVariant(-1));
        for(int i=0;i<l.size();i++)
        {
            if(!l.at(i).used)
                cb->addItem(l.at(i).name,l.at(i).index);
        }
        cb->setEditable(false);
        out = cb;
    }
    else if(index.column() == 2)
    {
        QComboBox *cb = new QComboBox(parent);
        cb->addItem(QString::fromUtf16(u"×"),QVariant(static_cast<int>(RfConfig::Multiply)));
        cb->addItem(QString::fromUtf16(u"÷"),QVariant(static_cast<int>(RfConfig::Divide)));
        cb->setEditable(false);
        out = cb;
    }
    else if(index.column() == 3)
    {
        QSpinBox *sb = new QSpinBox(parent);
        sb->setRange(1,1000000);
        out = sb;
    }
    else if(index.column() == 4)
    {
        auto l = dynamic_cast<const ClockTableModel*>(index.model())->getHwInfo();
        int id = index.model()->data(index.model()->index(index.row(),1),Qt::EditRole).toInt();
        QDoubleSpinBox *sb = new QDoubleSpinBox(parent);
        if(id >= 0 && id < l.size())
        {
            QString key = l.at(id).hwKey;
            SettingsStorage s(key,SettingsStorage::Hardware);
            double minFreq = s.get<double>(BC::Key::Clock::minFreq,0.0);
            double maxFreq = s.get<double>(BC::Key::Clock::maxFreq,1e7);

            //rescale range according to mult/div settings
            double factor = index.model()->data(index.model()->index(index.row(),3),Qt::EditRole).toDouble();
            RfConfig::MultOperation op = index.model()->data(index.model()->index(index.row(),2),Qt::EditRole).value<RfConfig::MultOperation>();
            if(op == RfConfig::Multiply)
                sb->setRange(minFreq*factor,maxFreq*factor);
            else
                sb->setRange(minFreq/factor,maxFreq/factor);
        }
        else
            sb->setRange(0.0,1e7);
        sb->setDecimals(6);
        out = sb;
    }

    return out;
}

void ClockTableDelegate::setEditorData(QWidget *editor, const QModelIndex &index) const
{
    if(index.column() == 1)
    {
        auto cb = dynamic_cast<QComboBox*>(editor);
        auto l = dynamic_cast<const ClockTableModel*>(index.model())->getHwInfo();
        int id = index.model()->data(index.model()->index(index.row(),1),Qt::EditRole).toInt();
        if(id >= 0 && id < l.size())
            cb->insertItem(0,l.at(id).name,l.at(id).index);

        cb->setCurrentIndex(0);
    }
    else if(index.column() == 2)
    {
        auto cb = dynamic_cast<QComboBox*>(editor);
        if(static_cast<RfConfig::MultOperation>(index.model()->data(index,Qt::EditRole).toInt()) == RfConfig::Multiply)
            cb->setCurrentIndex(0);
        else
            cb->setCurrentIndex(1);
    }
    else if(index.column() == 3)
    {
        auto sb = dynamic_cast<QSpinBox*>(editor);
        sb->setValue(index.model()->data(index,Qt::EditRole).toInt());
    }
    else if(index.column() == 4)
    {
        auto sb = dynamic_cast<QDoubleSpinBox*>(editor);
        sb->setValue(index.model()->data(index,Qt::EditRole).toDouble());
    }
}

void ClockTableDelegate::setModelData(QWidget *editor, QAbstractItemModel *model, const QModelIndex &index) const
{
    if(index.column() == 1 || index.column() == 2)
    {
        auto cb = dynamic_cast<QComboBox*>(editor);
        if(cb->currentData().toInt() < 0)
            model->setData(index,-1);
        else
            model->setData(index,cb->currentData());
    }
    else if(index.column() == 3)
    {
        auto sb = dynamic_cast<QSpinBox*>(editor);
        sb->interpretText();
        model->setData(index,sb->value());
    }
    else if(index.column() == 4)
    {
        auto sb = dynamic_cast<QDoubleSpinBox*>(editor);
        sb->interpretText();
        model->setData(index,sb->value());
    }
}

void ClockTableDelegate::updateEditorGeometry(QWidget *editor, const QStyleOptionViewItem &option, const QModelIndex &index) const
{
    Q_UNUSED(index)
    editor->setGeometry(option.rect);
}
