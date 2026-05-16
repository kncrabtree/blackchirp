#include <data/model/peaklistfilterproxymodel.h>

PeakListFilterProxyModel::PeakListFilterProxyModel(QObject *parent) :
    QSortFilterProxyModel(parent)
{
}

void PeakListFilterProxyModel::setStaticFilterEnabled(bool en)
{
    if(d_staticEnabled == en)
        return;
    beginFilterChange();
    d_staticEnabled = en;
    endFilterChange(QSortFilterProxyModel::Direction::Rows);
}

void PeakListFilterProxyModel::setMinFreq(double mhz)
{
    if(qFuzzyCompare(d_minFreq,mhz))
        return;
    beginFilterChange();
    d_minFreq = mhz;
    endFilterChange(QSortFilterProxyModel::Direction::Rows);
}

void PeakListFilterProxyModel::setMaxFreq(double mhz)
{
    if(qFuzzyCompare(d_maxFreq,mhz))
        return;
    beginFilterChange();
    d_maxFreq = mhz;
    endFilterChange(QSortFilterProxyModel::Direction::Rows);
}

void PeakListFilterProxyModel::setMinIntensity(double v)
{
    if(qFuzzyCompare(d_minInt,v))
        return;
    beginFilterChange();
    d_minInt = v;
    endFilterChange(QSortFilterProxyModel::Direction::Rows);
}

void PeakListFilterProxyModel::setMaxIntensity(double v)
{
    if(qFuzzyCompare(d_maxInt,v))
        return;
    beginFilterChange();
    d_maxInt = v;
    endFilterChange(QSortFilterProxyModel::Direction::Rows);
}

void PeakListFilterProxyModel::setViewSyncEnabled(bool en)
{
    if(d_viewSyncEnabled == en)
        return;
    beginFilterChange();
    d_viewSyncEnabled = en;
    endFilterChange(QSortFilterProxyModel::Direction::Rows);
}

void PeakListFilterProxyModel::setViewRange(double min, double max)
{
    if(qFuzzyCompare(d_viewMin,min) && qFuzzyCompare(d_viewMax,max))
        return;
    if(d_viewSyncEnabled)
    {
        beginFilterChange();
        d_viewMin = min;
        d_viewMax = max;
        endFilterChange(QSortFilterProxyModel::Direction::Rows);
    }
    else
    {
        d_viewMin = min;
        d_viewMax = max;
    }
}

bool PeakListFilterProxyModel::filterAcceptsRow(int sourceRow, const QModelIndex &sourceParent) const
{
    auto *src = sourceModel();
    if(!src)
        return true;

    const double freq = src->index(sourceRow,0,sourceParent).data(Qt::EditRole).toDouble();
    const double inten = src->index(sourceRow,1,sourceParent).data(Qt::EditRole).toDouble();

    if(d_staticEnabled)
    {
        if(freq < d_minFreq || freq > d_maxFreq)
            return false;
        if(inten < d_minInt || inten > d_maxInt)
            return false;
    }

    if(d_viewSyncEnabled)
    {
        if(freq < d_viewMin || freq > d_viewMax)
            return false;
    }

    return true;
}
