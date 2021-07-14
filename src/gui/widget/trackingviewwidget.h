#ifndef TRACKINGVIEWWIDGET_H
#define TRACKINGVIEWWIDGET_H

#include <QWidget>

#include <QList>
#include <QDateTime>

#include <qwt6/qwt_plot.h>
#include <data/storage/settingsstorage.h>
#include <data/storage/auxdatastorage.h>

class QGridLayout;
class TrackingPlot;
class BlackchirpPlotCurve;

namespace BC::Key {
static const QString trackingWidget("trackingWidget");
static const QString numPlots("numPlots");
static const QString viewonly("View");
static const QString plot("Plot");
}

class TrackingViewWidget : public QWidget, public SettingsStorage
{
    Q_OBJECT

public:
    explicit TrackingViewWidget(const QString name, QWidget *parent = 0, bool viewOnly = false);
    ~TrackingViewWidget();

    const QString d_name;

public slots:
    void initializeForExperiment();
    void pointUpdated(const AuxDataStorage::AuxDataMap m, const QDateTime t = QDateTime::currentDateTime());
    void moveCurveToPlot(BlackchirpPlotCurve *c, int newPlotIndex);
    void pushXAxis(int sourcePlotIndex);
    void autoScaleAll();

    void changeNumPlots();


private:
    QGridLayout *p_gridLayout = nullptr;
    QList<BlackchirpPlotCurve*> d_plotCurves;
    QList<TrackingPlot*> d_allPlots;
    bool d_viewMode;

    void addNewPlot();
    void configureGrid();
};

#endif // TRACKINGVIEWWIDGET_H
