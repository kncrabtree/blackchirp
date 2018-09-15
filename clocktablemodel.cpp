#include "clocktablemodel.h"

#include <QSettings>
#include <QComboBox>
#include <QSpinBox>
#include <QDoubleSpinBox>

ClockTableModel::ClockTableModel(QObject *parent) : QAbstractTableModel(parent)
{
    d_clockTypes = BlackChirp::allClockTypes();

}

void ClockTableModel::setConfig(const RfConfig c)
{
    d_rfConfig = c;
    d_hwInfo.clear();
    d_clockAssignments.clear();

    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(QString("clockManager"));
    int num = s.beginReadArray(QString("hwClocks"));
    for(int i=0; i<num; i++)
    {
        s.setArrayIndex(i);
        ClockHwInfo hw;
        hw.key = s.value(QString("key"),QString("")).toString();
        hw.output = s.value(QString("output"),0).toInt();
        hw.name = s.value(QString("name"),QString("%1-%2").arg(hw.key).arg(hw.output)).toString();
        hw.used = false;
        hw.index = d_hwInfo.size();
        if(!hw.key.isEmpty())
            d_hwInfo << hw;

        if(!d_rfConfig.getClocks().isEmpty())
        {
            for(auto it = d_rfConfig.getClocks().constBegin(); it!=d_rfConfig.getClocks().constEnd(); it++)
            {
                if(!hw.key.isEmpty() && it.value().hwKey == hw.key && it.value().output == hw.output)
                {
                    d_clockAssignments.insert(it.key(),hw.index);
                    break;
                }
            }
        }
    }
    s.endArray();
    s.endGroup();

    emit dataChanged(index(0,0),index(d_clockTypes.size(),5));
}

RfConfig ClockTableModel::getRfConfig() const
{
    return d_rfConfig;
}

void ClockTableModel::setCommonLo(bool b)
{
    d_rfConfig.setCommonLO(b);
    if(b)
    {
        d_rfConfig.setClockFreqInfo(BlackChirp::DownConversionLO,d_rfConfig.getClocks().value(BlackChirp::UpConversionLO));
        if(d_clockAssignments.contains(BlackChirp::UpConversionLO))
        {
            d_clockAssignments.insert(BlackChirp::DownConversionLO,d_clockAssignments.value(BlackChirp::UpConversionLO));
        }
    }
    else
    {
        d_rfConfig.setClockHwInfo(BlackChirp::DownConversionLO,QString(""),0);
        d_clockAssignments.remove(BlackChirp::DownConversionLO);
    }

    int downRow = d_clockTypes.indexOf(BlackChirp::DownConversionLO);
    emit dataChanged(index(downRow,1),index(downRow,5));

}



int ClockTableModel::rowCount(const QModelIndex &parent) const
{
    Q_UNUSED(parent)
    return d_clockTypes.size();
}

int ClockTableModel::columnCount(const QModelIndex &parent) const
{
    Q_UNUSED(parent)
    return 5;
}

QVariant ClockTableModel::data(const QModelIndex &index, int role) const
{
    if(index.row() > d_clockTypes.size())
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
            return BlackChirp::clockPrettyName(type);
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
                    .arg(BlackChirp::clockPrettyName(d_clockTypes.at(index.row())));
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

    if(index.row() > d_clockTypes.size())
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
            if(type == d_clockTypes.indexOf(BlackChirp::UpConversionLO) && d_rfConfig.commonLO())
                d_clockAssignments.insert(BlackChirp::DownConversionLO,value.toInt());

            d_rfConfig.setClockHwInfo(type,d_hwInfo.at(value.toInt()).key,d_hwInfo.at(value.toInt()).output);
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
    if(index.row() < d_clockTypes.size())
    {
        if(d_rfConfig.commonLO() && index.row() == d_clockTypes.indexOf(BlackChirp::DownConversionLO) && index.column() > 0)
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
            QString key = l.at(id).key;
            QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
            s.beginGroup(key);
            s.beginGroup(s.value(QString("subKey"),QString("virtual")).toString());
            double minFreq = s.value(QString("minFreqMHz"),0.0).toDouble();
            double maxFreq = s.value(QString("maxFreqMHz"),1e7).toDouble();
            s.endGroup();
            s.endGroup();
            sb->setRange(minFreq,maxFreq);
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
