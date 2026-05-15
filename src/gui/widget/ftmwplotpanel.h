#ifndef FTMWPLOTPANEL_H
#define FTMWPLOTPANEL_H

#include <QWidget>
#include <map>

#include <data/experiment/experiment.h>
#include <data/analysis/ftworker.h>

class QSpinBox;
class QDoubleSpinBox;
class QComboBox;
class QCheckBox;
class QPushButton;
class QGroupBox;
class QTableWidget;

class FtmwPlotPanel : public QWidget
{
    Q_OBJECT
public:
    enum MainPlotMode {
        Live,
        FT1,
        FT2,
        FT1_minus_FT2,
        FT2_minus_FT1,
        Upper_SideBand,
        Lower_SideBand,
        Both_SideBands
    };
    Q_ENUM(MainPlotMode)

    explicit FtmwPlotPanel(QWidget *parent = nullptr);
    void prepareForExperiment(const Experiment &e);
    void experimentComplete();
    void newBackup(int n);

    MainPlotMode mainPlotMode() const;
    int sbFrame() const;
    double sbMinFreq() const;
    double sbMaxFreq() const;
    FtWorker::DeconvolutionMethod dcMethod() const;

    int frame(int id) const;
    int segment(int id) const;
    int backup(int id) const;
    bool differential(int id) const;
    bool viewingBackup(int plotId) const;

signals:
    void mainPlotSettingChanged();
    void plotSettingChanged(int id);

private:
    QComboBox *p_mainPlotBox;

    QGroupBox *p_sbBox;
    QSpinBox *p_sbFrameBox;
    QDoubleSpinBox *p_sbMinBox, *p_sbMaxBox;
    QComboBox *p_sbAlgoBox;
    QPushButton *p_sbReprocessButton;

    struct PlotControls {
        QSpinBox *seg;
        QSpinBox *frame;
        QSpinBox *backup;
        QCheckBox *differential;
    };
    std::map<int,PlotControls> d_plotControls;

    void setMainPlotItemEnabled(MainPlotMode mode, bool enabled);
    QGroupBox *buildPlotSection(int id);
};

#endif // FTMWPLOTPANEL_H
