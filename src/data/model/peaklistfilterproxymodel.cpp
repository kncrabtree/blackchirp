#include <data/model/peaklistfilterproxymodel.h>

PeakListFilterProxyModel::PeakListFilterProxyModel(QObject *parent) :
    QSortFilterProxyModel(parent)
{
}

void PeakListFilterProxyModel::setStaticFilterEnabled(bool en)
{
    if(d_staticEnabled == en)
        return;
    d_staticEnabled = en;
    invalidateRowsFilter();
}

void PeakListFilterProxyModel::setMinFreq(double mhz)
{
    if(qFuzzyCompare(d_minFreq,mhz))
        return;
    d_minFreq = mhz;
    invalidateRowsFilter();
}

void PeakListFilterProxyModel::setMaxFreq(double mhz)
{
    if(qFuzzyCompare(d_maxFreq,mhz))
        return;
    d_maxFreq = mhz;
    invalidateRowsFilter();
}

void PeakListFilterProxyModel::setMinIntensity(double v)
{
    if(qFuzzyCompare(d_minInt,v))
        return;
    d_minInt = v;
    invalidateRowsFilter();
}

void PeakListFilterProxyModel::setViewSyncEnabled(bool en)
{
    if(d_viewSyncEnabled == en)
        return;
    d_viewSyncEnabled = en;
    invalidateRowsFilter();
}

void PeakListFilterProxyModel::setViewRange(double min, double max)
{
    if(qFuzzyCompare(d_viewMin,min) && qFuzzyCompare(d_viewMax,max))
        return;
    d_viewMin = min;
    d_viewMax = max;
    if(d_viewSyncEnabled)
        invalidateRowsFilter();
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
        if(inten < d_minInt)
            return false;
    }

    if(d_viewSyncEnabled)
    {
        if(freq < d_viewMin || freq > d_viewMax)
            return false;
    }

    return true;
}
