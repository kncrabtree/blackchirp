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
class SettingsTable;

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
    SettingsTable *p_table{nullptr};

    QComboBox *p_mainPlotBox{nullptr};

    QSpinBox *p_sbFrameBox{nullptr};
    QDoubleSpinBox *p_sbMinBox{nullptr};
    QDoubleSpinBox *p_sbMaxBox{nullptr};
    QComboBox *p_sbAlgoBox{nullptr};
    QPushButton *p_sbReprocessButton{nullptr};

    struct PlotControls {
        QSpinBox *seg;
        QSpinBox *frame;
        QSpinBox *backup;
        QCheckBox *differential;
    };
    std::map<int,PlotControls> d_plotControls;

    // Sideband row indices, assigned as the rows are appended; the
    // sideband group collapses as a unit for non-LO_Scan experiments.
    int d_rowSbFrame{-1};
    int d_rowSbMin{-1};
    int d_rowSbMax{-1};
    int d_rowSbAlgo{-1};

    void setMainPlotItemEnabled(MainPlotMode mode, bool enabled);
    void buildPlotControls(int id);
    void setSidebandRowsVisible(bool visible);
};

#endif // FTMWPLOTPANEL_H
