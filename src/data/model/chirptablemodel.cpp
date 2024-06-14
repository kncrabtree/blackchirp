#include <data/model/chirptablemodel.h>

#include <data/storage/settingsstorage.h>
#include <hardware/optional/chirpsource/awg.h>
#include <QDoubleSpinBox>
#include <QApplication>
#include <QCheckBox>

ChirpTableModel::ChirpTableModel(QObject *parent)
    : QAbstractTableModel(parent), SettingsStorage(BC::Key::ChirpTableModel::key),
      d_allIdentical(true), d_currentChirp(0)
{
    SettingsStorage s(BC::Key::hwKey(BC::Key::AWG::key,0)
                      ,SettingsStorage::Hardware);
    d_awgMin = s.get(BC::Key::AWG::min,0.0);
    d_awgMax = s.get(BC::Key::AWG::max,1000.0);

    using namespace BC::Key::ChirpTableModel;
    auto num = getArraySize(ctChirps);
    for(std::size_t i=0; i<num; ++i)
    {
        auto ci = getArrayValue(ctChirps,i,chirpIndex,0);
        auto si = getArrayValue(ctChirps,i,segIndex,0);
        auto st = getArrayValue(ctChirps,i,start,d_awgMin);
        auto en = getArrayValue(ctChirps,i,end,d_awgMax);
        auto dur = getArrayValue(ctChirps,i,duration,0.5);
        auto e = getArrayValue(ctChirps,i,empty,true);

        if(d_chirpList.size() < ci+1)
            d_chirpList.resize(ci+1);

        if(d_chirpList.at(ci).size() < si+1)
            d_chirpList[ci].resize(si+1);

        d_chirpList[ci][si] = {st,en,dur,(en-st)/dur,e};
    }
}

ChirpTableModel::~ChirpTableModel()
{
    using namespace BC::Key::ChirpTableModel;
    setArray(ctChirps,{});
    for(int ci=0; ci<d_chirpList.size(); ++ci)
    {
        auto sl = d_chirpList.at(ci);
        for(int si=0; si<sl.size(); ++si)
        {
            auto &seg = sl.at(si);
            appendArrayMap(ctChirps,{
                               {chirpIndex,ci},
                               {segIndex,si},
                               {start,seg.startFreqMHz},
                               {end,seg.endFreqMHz},
                               {duration,seg.durationUs},
                               {empty,seg.empty}
                           });
        }
    }
}



int ChirpTableModel::rowCount(const QModelIndex &parent) const
{
    Q_UNUSED(parent)
    if(d_currentChirp >= d_chirpList.size())
        return 0;

    return d_chirpList.at(d_currentChirp).size();
}

int ChirpTableModel::columnCount(const QModelIndex &parent) const
{
    Q_UNUSED(parent)
    return 6;
}

QVariant ChirpTableModel::data(const QModelIndex &index, int role) const
{
    auto segmentList = d_chirpList.at(d_currentChirp);
    if(index.row() >= segmentList.size())
        return QVariant();

    if(role == Qt::TextAlignmentRole)
        return Qt::AlignCenter;

    bool empty = segmentList.at(index.row()).empty;

    if(role == Qt::DisplayRole)
    {
        switch(index.column()) {
        case 0:
            if(empty)
                return QString("Empty");
            else
            {
                double chirpFreq = p_rfConfig->calculateChirpFreq(segmentList.at(index.row()).startFreqMHz);
                return QString::number(chirpFreq,'f',3);
            }
            break;
        case 1:
            if(empty)
                return QString("Empty");
            else
            {
                double chirpFreq = p_rfConfig->calculateChirpFreq(segmentList.at(index.row()).endFreqMHz);
                return QString::number(chirpFreq,'f',3);
            }
            break;
        case 2:
            return QString::number(segmentList.at(index.row()).durationUs*1e3,'f',1);
            break;
        case 3:
            if(empty)
                return QString("Yes");
            else
                return QString("No");
            break;
        case 4:
            if(empty)
                return QString("Empty");
            else
                return QString::number(segmentList.at(index.row()).startFreqMHz,'f',3);
        case 5:
            if(empty)
                return QString("Empty");
            else
                return QString::number(segmentList.at(index.row()).endFreqMHz,'f',3);
        default:
            return QVariant();
            break;
        }
    }
    else if(role == Qt::EditRole)
    {
        switch(index.column()) {
        case 0:
            return  p_rfConfig->calculateChirpFreq(segmentList.at(index.row()).startFreqMHz);
            break;
        case 1:
            return  p_rfConfig->calculateChirpFreq(segmentList.at(index.row()).endFreqMHz);
            break;
        case 2:
            return segmentList.at(index.row()).durationUs*1e3;
            break;
        case 3:
            return segmentList.at(index.row()).empty;
            break;
        case 4:
            return segmentList.at(index.row()).startFreqMHz;
            break;
        case 5:
            return segmentList.at(index.row()).endFreqMHz;
            break;
        default:
            return QVariant();
            break;
        }
    }

    return QVariant();
}

QVariant ChirpTableModel::headerData(int section, Qt::Orientation orientation, int role) const
{
    if(role == Qt::DisplayRole)
    {
        if(orientation == Qt::Vertical)
            return section + 1;
        else
        {
            switch(section) {
            case 0:
                return QString("Chirp Start (MHz)");
                break;
            case 1:
                return QString("Chirp End (MHz)");
                break;
            case 2:
                return QString("Duration (ns)");
                break;
            case 3:
                return QString("Empty?");
                break;
            case 4:
                return QString("AWG Start (MHz)");
                break;
            case 5:
                return QString("AWG End (MHz)");
                break;
            default:
                return QVariant();
                break;
            }
        }
    }
    else if(role == Qt::ToolTipRole)
    {
        if(orientation == Qt::Vertical)
            return QVariant();

        switch(section) {
        case 0:
            return QString("Starting frequency for the chirp segment (in MHz).");
            break;
        case 1:
            return QString("Ending frequency for the chirp segment (in MHz).");
            break;
        case 2:
            return QString("Duration of the chirp (in ns)");
            break;
        case 3:
            return QString("An empty segment makes a gap in the chirp.");
            break;
        case 4:
            return QString("Starting AWG frequency for the chirp segment (in MHz)");
            break;
        case 5:
            return QString("Ending AWG frequency for the chirp segment (in MHz)");
            break;
        default:
            return QVariant();
            break;
        }

    }

    return QVariant();
}

bool ChirpTableModel::setData(const QModelIndex &index, const QVariant &value, int role)
{
    if(role != Qt::EditRole)
        return false;

    if(index.row() >= d_chirpList.at(d_currentChirp).size() || index.column() > 5)
        return false;

    int ll = d_currentChirp;
    int ul = d_currentChirp+1;
    if(d_allIdentical)
    {
        ll = 0;
        ul = d_chirpList.size();
    }

    for(int i=ll; i<ul; i++)
    {
        switch (index.column()) {
        case 0:
            d_chirpList[i][index.row()].startFreqMHz = p_rfConfig->calculateAwgFreq(value.toDouble());
            break;
        case 1:
            d_chirpList[i][index.row()].endFreqMHz = p_rfConfig->calculateAwgFreq(value.toDouble());
            break;
        case 2:
            d_chirpList[i][index.row()].durationUs = value.toDouble()/1e3;
            break;
        case 3:
            d_chirpList[i][index.row()].empty = value.toBool();
            if(d_chirpList.at(i).at(index.row()).empty)
            {
                d_chirpList[i][index.row()].startFreqMHz = 0.0;
                d_chirpList[i][index.row()].endFreqMHz = 0.0;
            }
            else
            {
                d_chirpList[i][index.row()].startFreqMHz = d_awgMin;
                d_chirpList[i][index.row()].endFreqMHz = d_awgMax;
            }
            break;
        case 4:
            d_chirpList[i][index.row()].startFreqMHz = value.toDouble();
            break;
        case 5:
            d_chirpList[i][index.row()].endFreqMHz = value.toDouble();
            break;
        default:
            return false;
            break;
        }

        d_chirpList[i][index.row()].alphaUs = (d_chirpList.at(i).at(index.row()).endFreqMHz - d_chirpList.at(i).at(index.row()).startFreqMHz)/d_chirpList.at(i).at(index.row()).durationUs;
    }

    emit dataChanged(index,index);
    emit modelChanged();
    return true;
}

bool ChirpTableModel::removeRows(int row, int count, const QModelIndex &parent)
{
    if(d_chirpList.isEmpty())
        return false;

    if(row < 0 || row+count > d_chirpList.at(d_currentChirp).size() || d_chirpList.at(d_currentChirp).isEmpty())
        return false;

    int ll = d_currentChirp;
    int ul = d_currentChirp+1;
    if(d_allIdentical)
    {
        ll = 0;
        ul = d_chirpList.size();
    }
    for(int j=ll; j<ul; j++)
    {
        for(int i=0; i<count; i++)
        {
            if(j == d_currentChirp)
                beginRemoveRows(parent,row,row+count-1);

            d_chirpList[j].removeAt(row);

            if(j == d_currentChirp)
                endRemoveRows();
        }
    }

    emit modelChanged();
    return true;
}

Qt::ItemFlags ChirpTableModel::flags(const QModelIndex &index) const
{
    if(index.row() < d_chirpList.at(d_currentChirp).size())
    {
        if(!d_chirpList.at(d_currentChirp).at(index.row()).empty || index.column() == 2 || index.column() == 3)
            return Qt::ItemIsEnabled|Qt::ItemIsSelectable|Qt::ItemIsEditable;
        else
            return Qt::ItemIsSelectable;
    }

    return Qt::ItemIsEnabled|Qt::ItemIsSelectable;
}

void ChirpTableModel::addSegment(double start, double end, double dur, int pos, bool empty)
{

    if(start < 0.0)
        start = d_awgMin;
    if(end < 0.0)
        end = d_awgMax;

    double startFreq = qBound(d_awgMin,start,d_awgMax);
    double endFreq = qBound(d_awgMin,end,d_awgMax);

    ChirpConfig::ChirpSegment cs{startFreq,endFreq,dur,(end-start)/dur,empty};

    if(d_chirpList.isEmpty())
    {
        d_currentChirp = 0;
        QVector<ChirpConfig::ChirpSegment> l;
        l << cs;
        beginInsertRows(QModelIndex(),0,0);
        d_chirpList << l;
        endInsertRows();
        emit modelChanged();
        return;
    }

    int ll = d_currentChirp;
    int ul = d_currentChirp+1;
    if(d_allIdentical)
    {
        ll = 0;
        ul = d_chirpList.size();
    }

    if(pos < 0 || pos >= d_chirpList.at(d_currentChirp).size())
    {
        for(int i=ll; i<ul; i++)
        {
            if(i == d_currentChirp)
                beginInsertRows(QModelIndex(),d_chirpList.at(d_currentChirp).size(),d_chirpList.at(d_currentChirp).size());

            d_chirpList[i].append(cs);

            if(i == d_currentChirp)
                endInsertRows();
        }
    }
    else
    {

        for(int i=ll; i<ul; i++)
        {
            if(i == d_currentChirp)
                beginInsertRows(QModelIndex(),pos,pos);

            d_chirpList[i].insert(pos,cs);

            if(i == d_currentChirp)
                endInsertRows();
        }

    }

    emit modelChanged();
}

void ChirpTableModel::moveSegments(int first, int last, int delta)
{
    //make sure all movement is within valid ranges
    if(first + delta < 0 || last + delta >= d_chirpList.at(d_currentChirp).size())
        return;

    int ll = d_currentChirp;
    int ul = d_currentChirp+1;
    if(d_allIdentical)
    {
        ll = 0;
        ul = d_chirpList.size();
    }

    //this bit of code is not intuitive! read docs on QAbstractItemModel::beginMoveRows() carefully!
    if(delta>0)
    {
        if(!beginMoveRows(QModelIndex(),first,last,QModelIndex(),last+2))
            return;
    }
    else
    {
        if(!beginMoveRows(QModelIndex(),first,last,QModelIndex(),first-1))
            return;
    }

    for(int j=ul; j<ll; j++)
    {
        auto chunk = d_chirpList.at(j).mid(first,last-first+1);

        //remove selected rows
        for(int i=0; i<last-first+1; i++)
            d_chirpList[j].removeAt(first);

        //insert rows at their new location
        for(int i = chunk.size(); i>0; i--)
        {
            if(delta>0)
                d_chirpList[j].insert(first+1,chunk.at(i-1));
            else
                d_chirpList[j].insert(first-1,chunk.at(i-1));
        }
    }
    endMoveRows();

    emit modelChanged();
}

void ChirpTableModel::removeSegments(QList<int> rows)
{
    std::sort(rows.begin(),rows.end());
    for(int i=rows.size(); i>0; i--)
        removeRows(rows.at(i-1),1,QModelIndex());
}

double ChirpTableModel::calculateAwgFrequency(double f) const
{
    return p_rfConfig->calculateAwgFreq(f);
}

double ChirpTableModel::calculateChirpFrequency(double f) const
{
    return p_rfConfig->calculateChirpFreq(f);
}

QVector<QVector<ChirpConfig::ChirpSegment> > ChirpTableModel::chirpList() const
{
    return d_chirpList;
}

void ChirpTableModel::initialize(RfConfig *p)
{
    p_rfConfig = p;
    p_rfConfig->d_chirpConfig.setChirpList(d_chirpList);
}

void ChirpTableModel::setFromRfConfig(RfConfig *p)
{
    p_rfConfig = p;
    removeRows(0,rowCount(QModelIndex()),QModelIndex());
    d_chirpList.clear();
    d_currentChirp = 0;
    auto cl = p_rfConfig->d_chirpConfig.chirpList();
    if(!cl.isEmpty())
    {
        auto s = cl.constFirst().size();
        beginInsertRows(QModelIndex(),0,s-1);
        d_chirpList = cl;
        endInsertRows();
        emit modelChanged();
    }
}

void ChirpTableModel::setCurrentChirp(int i)
{
    beginRemoveRows(QModelIndex(),0,d_chirpList.at(d_currentChirp).size()-1);
    endRemoveRows();
    d_currentChirp = i;
    beginInsertRows(QModelIndex(),0,d_chirpList.at(d_currentChirp).size()-1);
    endInsertRows();
    emit modelChanged();
}

void ChirpTableModel::setNumChirps(int num)
{
    if(num > d_chirpList.size())
    {
        if(!d_chirpList.isEmpty())
        {
            for(int i=d_chirpList.size(); i<num; i++)
                d_chirpList.append(d_chirpList.constFirst());
        }
        else
        {
            for(int i=d_chirpList.size(); i<num; i++)
                d_chirpList.append(QVector<ChirpConfig::ChirpSegment>());
        }
    }
    else if(num < d_chirpList.size())
    {
        while(d_chirpList.size() > num)
            d_chirpList.removeLast();
    }
}


ChirpDoubleSpinBoxDelegate::ChirpDoubleSpinBoxDelegate(QObject *parent) : QStyledItemDelegate(parent)
{
}

QWidget *ChirpDoubleSpinBoxDelegate::createEditor(QWidget *parent, const QStyleOptionViewItem &option, const QModelIndex &index) const
{
    Q_UNUSED(option)

    QDoubleSpinBox *editor = new QDoubleSpinBox(parent);
    QWidget *out = editor;

    auto m = static_cast<const ChirpTableModel*>(index.model());


    double chirpMin = m->calculateChirpFrequency(m->awgMin());
    double chirpMax = m->calculateChirpFrequency(m->awgMax());
    if(chirpMin > chirpMax)
        qSwap(chirpMin,chirpMax);

    bool empty = index.model()->data(index.model()->index(index.row(),3),Qt::EditRole).toBool();

    switch(index.column())
    {
    case 0:
    case 1:
        if(!empty)
        {
            editor->setRange(chirpMin,chirpMax);
            editor->setDecimals(3);
            editor->setEnabled(true);
        }
        else
        {
            editor->setMinimum(0.0);
            editor->setMaximum(0.0);
            editor->setDecimals(3);
            editor->setEnabled(false);
            editor->setSpecialValueText(QString("Empty"));
        }
        break;
    case 2:
        editor->setMinimum(0.1);
        editor->setMaximum(100000.0);
        editor->setSingleStep(10.0);
        editor->setDecimals(1);
        break;
    case 3:
        out = new QCheckBox(parent);
        break;
    case 4:
    case 5:
        if(!empty)
        {
            editor->setRange(m->awgMin(),m->awgMax());
            editor->setDecimals(3);
            editor->setEnabled(true);
        }
        else
        {
            editor->setMinimum(0.0);
            editor->setMaximum(0.0);
            editor->setDecimals(3);
            editor->setEnabled(false);
            editor->setSpecialValueText(QString("Empty"));
        }
        break;
    default:
        break;
    }

    return out;
}

void ChirpDoubleSpinBoxDelegate::setEditorData(QWidget *editor, const QModelIndex &index) const
{
    if(index.column() != 3)
    {
        double val = index.model()->data(index, Qt::EditRole).toDouble();

        static_cast<QDoubleSpinBox*>(editor)->setValue(val);
    }
    else
    {
        bool empty = index.model()->data(index, Qt::EditRole).toBool();

        static_cast<QCheckBox*>(editor)->setChecked(empty);
    }
}

void ChirpDoubleSpinBoxDelegate::setModelData(QWidget *editor, QAbstractItemModel *model, const QModelIndex &index) const
{
    if(index.column() != 3)
    {
        QDoubleSpinBox *sb = static_cast<QDoubleSpinBox*>(editor);
        sb->interpretText();
        model->setData(index,sb->value(),Qt::EditRole);
    }
    else
    {
        QCheckBox *cb = static_cast<QCheckBox*>(editor);
        model->setData(index,cb->isChecked(),Qt::EditRole);
    }
}

void ChirpDoubleSpinBoxDelegate::updateEditorGeometry(QWidget *editor, const QStyleOptionViewItem &option, const QModelIndex &index) const
{
    Q_UNUSED(index)
    editor->setGeometry(option.rect);
}
