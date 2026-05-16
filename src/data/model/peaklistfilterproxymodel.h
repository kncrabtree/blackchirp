#ifndef PEAKLISTFILTERPROXYMODEL_H
#define PEAKLISTFILTERPROXYMODEL_H

#include <QSortFilterProxyModel>

#include <limits>

/// \brief Display filter over PeakListModel for the Peak Find table.
///
/// Filters rows by frequency range and minimum intensity (the "static"
/// filters) and, independently, by the main FT plot's currently visible
/// x range (the "in view" sync filter). These are *display* filters: they
/// hide rows from the table only and never alter the underlying peak
/// list, the search bounds, or the peaks drawn on the plot. They are
/// distinct from the Peak Finding Options dialog's min/max frequency,
/// which bound the *search* itself.
///
/// Source columns are PeakListModel's: column 0 = frequency (MHz),
/// column 1 = intensity, both read via \c Qt::EditRole as doubles.
class PeakListFilterProxyModel : public QSortFilterProxyModel
{
    Q_OBJECT
public:
    explicit PeakListFilterProxyModel(QObject *parent = nullptr);

public slots:
    /// \brief Master enable for the static (frequency/intensity) filters.
    ///
    /// When disabled, the frequency-range and minimum-intensity bounds
    /// are ignored regardless of their values; the view-sync filter is
    /// unaffected.
    void setStaticFilterEnabled(bool en);

    /// \brief Lower frequency bound in MHz. Pass \c -infinity for no lower bound.
    void setMinFreq(double mhz);

    /// \brief Upper frequency bound in MHz. Pass \c +infinity for no upper bound.
    void setMaxFreq(double mhz);

    /// \brief Minimum intensity. Pass \c -infinity (or any value at or
    ///        below zero, since intensities are non-negative) for no bound.
    void setMinIntensity(double v);

    /// \brief Maximum intensity. Pass \c +infinity for no upper bound.
    void setMaxIntensity(double v);

    /// \brief Enables or disables the "only peaks in the main-plot view" filter.
    void setViewSyncEnabled(bool en);

    /// \brief Sets the visible x range (MHz) used by the view-sync filter.
    ///
    /// Fed continuously by the host from the main FT plot's
    /// ZoomPanPlot::visibleXRangeChanged signal; only affects displayed
    /// rows while \c setViewSyncEnabled(true).
    void setViewRange(double min, double max);

protected:
    bool filterAcceptsRow(int sourceRow, const QModelIndex &sourceParent) const override;

private:
    /// \brief Re-evaluates the row filter across Qt versions.
    ///
    /// Qt 6.10 introduced \c endFilterChange(Direction) and the \c
    /// Direction enum; \c beginFilterChange() arrived in 6.9 and the
    /// older \c invalidateRowsFilter() is deprecated from 6.13. This
    /// wrapper uses the begin/end pair where available and falls back
    /// to \c invalidateRowsFilter() on the 6.4 minimum target.
    void reapplyRowFilter();

    static constexpr double s_inf = std::numeric_limits<double>::infinity();

    bool d_staticEnabled{false};
    double d_minFreq{-s_inf};
    double d_maxFreq{s_inf};
    double d_minInt{-s_inf};
    double d_maxInt{s_inf};

    bool d_viewSyncEnabled{false};
    double d_viewMin{-s_inf};
    double d_viewMax{s_inf};
};

#endif // PEAKLISTFILTERPROXYMODEL_H
