#ifndef QUICKEXPTDIALOG_H
#define QUICKEXPTDIALOG_H

#include <QDialog>

#include <src/data/experiment/experiment.h>

namespace Ui {
class QuickExptDialog;
}

class QuickExptDialog : public QDialog
{
    Q_OBJECT

public:
    explicit QuickExptDialog(Experiment e, QWidget *parent = 0);
    ~QuickExptDialog();

    int configureResult() const { return d_configureResult; }
    bool sleepWhenDone() const;

private:
    Ui::QuickExptDialog *ui;

    const int d_configureResult = 17;
};

#endif // QUICKEXPTDIALOG_H
