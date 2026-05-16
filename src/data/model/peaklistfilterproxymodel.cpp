#include <data/model/peaklistfilterproxymodel.h>

PeakListFilterProxyModel::PeakListFilterProxyModel(QObject *parent) :
    QSortFilterProxyModel(parent)
{
}

void PeakListFilterProxyModel::reapplyRowFilter()
{
#if QT_VERSION >= QT_VERSION_CHECK(6, 10, 0)
    beginFilterChange();
    endFilterChange(QSortFilterProxyModel::Direction::Rows);
#else
    invalidateRowsFilter();
#endif
}

void PeakListFilterProxyModel::setStaticFilterEnabled(bool en)
{
    if(d_staticEnabled == en)
        return;
    d_staticEnabled = en;
    reapplyRowFilter();
}

void PeakListFilterProxyModel::setMinFreq(double mhz)
{
    if(qFuzzyCompare(d_minFreq,mhz))
        return;
    d_minFreq = mhz;
    reapplyRowFilter();
}

void PeakListFilterProxyModel::setMaxFreq(double mhz)
{
    if(qFuzzyCompare(d_maxFreq,mhz))
        return;
    d_maxFreq = mhz;
    reapplyRowFilter();
}

void PeakListFilterProxyModel::setMinIntensity(double v)
{
    if(qFuzzyCompare(d_minInt,v))
        return;
    d_minInt = v;
    reapplyRowFilter();
}

void PeakListFilterProxyModel::setMaxIntensity(double v)
{
    if(qFuzzyCompare(d_maxInt,v))
        return;
    d_maxInt = v;
    reapplyRowFilter();
}

void PeakListFilterProxyModel::setViewSyncEnabled(bool en)
{
    if(d_viewSyncEnabled == en)
        return;
    d_viewSyncEnabled = en;
    reapplyRowFilter();
}

void PeakListFilterProxyModel::setViewRange(double min, double max)
{
    if(qFuzzyCompare(d_viewMin,min) && qFuzzyCompare(d_viewMax,max))
        return;
    d_viewMin = min;
    d_viewMax = max;
    if(d_viewSyncEnabled)
        reapplyRowFilter();
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
