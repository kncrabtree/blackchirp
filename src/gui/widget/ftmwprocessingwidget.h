#ifndef FTMWPROCESSINGWIDGET_H
#define FTMWPROCESSINGWIDGET_H

#include <QWidget>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QCheckBox>
#include <QRadioButton>

#include <src/data/experiment/experiment.h>

class FtmwProcessingWidget : public QWidget
{
    Q_OBJECT
public:
    explicit FtmwProcessingWidget(QWidget *parent = 0);

signals:
    void settingsUpdated(FtWorker::FidProcessingSettings);

public slots:
    void prepareForExperient(const Experiment e);
    void applySettings(FtWorker::FidProcessingSettings s);
    void readSettings();

private:
    QDoubleSpinBox *p_startBox, *p_endBox, *p_autoScaleIgnoreBox;
    QSpinBox *p_zeroPadBox;
    QCheckBox *p_removeDCBox;
    QMap<BlackChirp::FtWindowFunction,QString> d_windowTypes;
    QMap<BlackChirp::FtWindowFunction,QRadioButton*> d_windowButtons;
    QMap<BlackChirp::FtPlotUnits,QString> d_ftUnits;
    QMap<BlackChirp::FtPlotUnits,QRadioButton*> d_unitsButtons;


};

#endif // FTMWPROCESSINGWIDGET_H
