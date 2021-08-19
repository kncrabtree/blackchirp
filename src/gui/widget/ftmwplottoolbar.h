#ifndef FTMWPLOTTOOLBAR_H
#define FTMWPLOTTOOLBAR_H

#include <QToolBar>
#include <data/experiment/experiment.h>

class SpinBoxWidgetAction;
class DoubleSpinBoxWidgetAction;
template<typename T>
class EnumComboBoxWidgetAction;

class FtmwPlotToolBar : public QToolBar
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

    FtmwPlotToolBar(QWidget *parent = nullptr);
    void prepareForExperiment(const Experiment &e);
    void experimentComplete();
    void newBackup(int n);

    MainPlotMode mainPlotMode() const;
    int mainPlotFollow() const;
    double sbMinFreq() const;
    double sbMaxFreq() const;

    int frame(int id) const;
    int segment(int id) const;
    int backup(int id) const;
    bool viewingBackup(int plotId) const;

signals:
    void mainPlotSettingChanged();
    void plotSettingChanged(int id);

private:
    EnumComboBoxWidgetAction<MainPlotMode> *p_mainPlotBox;
    SpinBoxWidgetAction *p_followBox;
    DoubleSpinBoxWidgetAction *p_sbMinBox, *p_sbMaxBox;

    std::map<int,SpinBoxWidgetAction*> d_seg;
    std::map<int,SpinBoxWidgetAction*> d_frame;
    std::map<int,SpinBoxWidgetAction*> d_backup;
};

#endif // FTMWPLOTTOOLBAR_H
