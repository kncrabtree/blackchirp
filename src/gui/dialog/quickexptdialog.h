#ifndef QUICKEXPTDIALOG_H
#define QUICKEXPTDIALOG_H

#include <QDialog>
#include <memory>

class Experiment;
class ExperimentSummaryWidget;
class QLabel;
class QSpinBox;
class QCheckBox;
class QFormLayout;
class QPushButton;

namespace Ui {
class QuickExptDialog;
}

class QuickExptDialog : public QDialog
{
    Q_OBJECT
public:
    enum QeResult {
        New=1000,
        Configure,
        Start,
    };
    explicit QuickExptDialog(QWidget *parent = nullptr);

    std::map<QString,bool,std::less<>> getOptHwSettings() const;
    int exptNumber() const;

private slots:
    void loadExperiment(int num);

private:
    const int d_configureResult = 17;
    QSpinBox *p_expSpinBox;
    QLabel *p_warningLabel;
    QFormLayout *p_hwLayout;
    QPushButton *p_cfgButton, *p_startButton;
    ExperimentSummaryWidget *p_esw;
    std::map<QString,QString,std::less<>> d_hardware;
    std::map<QString,QCheckBox*,std::less<>> d_hwBoxes;
};

#endif // QUICKEXPTDIALOG_H
