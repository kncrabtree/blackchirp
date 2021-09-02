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
class BlackchirpPlotCurveBase;

namespace BC::Key {
static const QString auxDataWidget{"AuxDataWidget"};
static const QString rollingDataWidget{"RollingDataWidget"};
static const QString numPlots{"numPlots"};
static const QString viewonly{"View"};
static const QString plot{"Plot"};
static const QString history{"historyHours"};
}

class AuxDataViewWidget : public QWidget, public SettingsStorage
{
    Q_OBJECT

public:
    explicit AuxDataViewWidget(const QString name, QWidget *parent = nullptr, bool viewOnly = false);
    virtual ~AuxDataViewWidget();

    const QString d_name;
    int numPlots() const { return d_allPlots.size(); }

public slots:
    void initializeForExperiment();
    virtual void pointUpdated(const AuxDataStorage::AuxDataMap m, const QDateTime t = QDateTime::currentDateTime());
    void moveCurveToPlot(BlackchirpPlotCurve *c, int newPlotIndex);
    void pushXAxis(int sourcePlotIndex);
    void autoScaleAll();

    void changeNumPlots(int newNum);
    virtual void purgeOldPoints(BlackchirpPlotCurve *c) { Q_UNUSED(c) }

protected:
    TrackingPlot* getPlot(int i) { return d_allPlots.at(i); }
    BlackchirpPlotCurve* getCurve(int i) { return d_plotCurves.at(i); }
    int numCurves() const { return d_plotCurves.size(); }
    QVector<BlackchirpPlotCurve*> d_plotCurves;


private:
    QGridLayout *p_gridLayout = nullptr;
    QVector<TrackingPlot*> d_allPlots;
    bool d_viewMode;

    void addNewPlot();
    void configureGrid();
};

class RollingDataWidget : public AuxDataViewWidget
{
    Q_OBJECT
public:
    explicit RollingDataWidget(const QString name, QWidget *parent = nullptr);
    ~RollingDataWidget() {};

    void pointUpdated(const AuxDataStorage::AuxDataMap m, const QDateTime dt = QDateTime::currentDateTime()) override;
    void purgeOldPoints(BlackchirpPlotCurve *c) override;
    void setHistoryDuration(int d);
    int historyDuration() const { return d_historyDuration; }

private:
    int d_historyDuration{12};
};

#endif // TRACKINGVIEWWIDGET_H
