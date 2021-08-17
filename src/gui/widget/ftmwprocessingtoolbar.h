#ifndef FTMWPROCESSINGTOOLBAR_H
#define FTMWPROCESSINGTOOLBAR_H

#include <QToolBar>

#include <data/storage/settingsstorage.h>
#include <data/experiment/experiment.h>
#include <data/analysis/ftworker.h>

class QSpinBox;
class QDoubleSpinBox;
class QToolButton;
class QComboBox;
class SpinBoxWidgetAction;
class DoubleSpinBoxWidgetAction;
template<typename T>
class EnumComboBoxWidgetAction;

namespace BC::Key {
static const QString ftmwProcWidget("ftmwProcessingWidget");
static const QString fidStart("startUs");
static const QString fidEnd("endUs");
static const QString zeroPad("zeroPad");
static const QString removeDC("removeDC");
static const QString ftUnits("ftUnits");
static const QString autoscaleIgnore("autoscaleIgnoreMHz");
static const QString ftWinf("windowFunction");
}

class FtmwProcessingToolBar : public QToolBar, public SettingsStorage
{
    Q_OBJECT
public:
    explicit FtmwProcessingToolBar(QWidget *parent = 0);
    ~FtmwProcessingToolBar();
    FtWorker::FidProcessingSettings getSettings();

signals:
    void settingsUpdated(FtWorker::FidProcessingSettings);

public slots:
    void prepareForExperient(const Experiment &e);
    void readSettings();

private:
    DoubleSpinBoxWidgetAction *p_startBox, *p_endBox, *p_autoScaleIgnoreBox;
    SpinBoxWidgetAction *p_zeroPadBox;
    QAction *p_removeDCBox;
    EnumComboBoxWidgetAction<FtWorker::FtUnits> *p_unitsBox;
    EnumComboBoxWidgetAction<FtWorker::FtWindowFunction> *p_winfBox;



};



#endif // FTMWPROCESSINGTOOLBAR_H
