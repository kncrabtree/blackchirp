#ifndef LIFDISPLAYWIDGET_H
#define LIFDISPLAYWIDGET_H

#include <QWidget>

#include <QVector>
#include <memory.h>

#include <data/experiment/experiment.h>
#include <data/storage/settingsstorage.h>

class LifSlicePlot;
class LifTracePlot;
class LifSpectrogramPlot;
class LifProcessingWidget;
class QSpinBox;

namespace BC::Key::LifDW {
static const QString lifDwKey{"LifDisplayWidget"};
static const QString refresh{"refreshIntervalMs"};
static const QString lifSpectrumPlot{"lifSpectrumPlot"};
static const QString lifTimePlot{"lifTimePlot"};
}

class LifDisplayWidget : public QWidget, public SettingsStorage
{
    Q_OBJECT

public:
    explicit LifDisplayWidget(QWidget *parent = 0);
    ~LifDisplayWidget();

    QVector<QPointF> laserSlice(int delayIndex) const;
    QVector<QPointF> delaySlice(int laserIndex) const;

public slots:
    void prepareForExperiment(const Experiment &e);
    void experimentComplete();
    void updatePoint();
    void updatePlots();

    void changeLaserSlice(int di);
    void changeDelaySlice(int li);
    void reprocess();
    void resetProc();
    void saveProc();

private:
    std::shared_ptr<LifStorage> ps_lifStorage;
    bool d_delayReverse{false}, d_laserReverse{false};
    QVector<double> d_currentIntegratedData;

    LifSlicePlot *p_delaySlicePlot, *p_laserSlicePlot;
    LifTracePlot *p_lifTracePlot;
    LifSpectrogramPlot *p_spectrogramPlot;
    LifProcessingWidget *p_procWidget;
    
    QSpinBox *p_refreshBox;

    QString d_dString;
    QString d_lString;
    int d_lDec{2};
    int d_refreshTimerId{-1};


    
    // QObject interface
protected:
    void timerEvent(QTimerEvent *event) override;
};

#endif // LIFDISPLAYWIDGET_H
