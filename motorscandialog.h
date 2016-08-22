#ifndef MOTORSCANDIALOG_H
#define MOTORSCANDIALOG_H

#include <QDialog>

#include "motorscan.h"

namespace Ui {
class MotorScanDialog;
}

class MotorScanDialog : public QDialog
{
    Q_OBJECT

public:
    explicit MotorScanDialog(QWidget *parent = 0);
    ~MotorScanDialog();

    void setFromMotorScan(MotorScan ms);
    MotorScan toMotorScan();

private:
    Ui::MotorScanDialog *ui;

    void validateBoxes();


    // QDialog interface
public slots:
    void accept();
};

#endif // MOTORSCANDIALOG_H
