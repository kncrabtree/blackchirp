#ifndef EXPORTBATCHDIALOG_H
#define EXPORTBATCHDIALOG_H

#include <QDialog>

namespace Ui {
class ExportBatchDialog;
}

class ExportBatchDialog : public QDialog
{
    Q_OBJECT

public:
    explicit ExportBatchDialog(QWidget *parent = 0);
    ~ExportBatchDialog();

public slots:
    void selectDirectory();
    // QDialog interface
    void accept();
    void checkComplete();

private:
    Ui::ExportBatchDialog *ui;

};

#endif // EXPORTBATCHDIALOG_H
