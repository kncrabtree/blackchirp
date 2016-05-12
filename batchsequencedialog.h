#ifndef BATCHSEQUENCEDIALOG_H
#define BATCHSEQUENCEDIALOG_H

#include <QDialog>

namespace Ui {
class BatchSequenceDialog;
}

class BatchSequenceDialog : public QDialog
{
    Q_OBJECT

public:
    explicit BatchSequenceDialog(QWidget *parent = 0);
    ~BatchSequenceDialog();

    void setQuickExptEnabled(bool en);

    int configureCode() const { return d_configureCode; }
    int quickCode() const { return d_quickCode; }

    int numExperiments() const;
    int interval() const;
    bool autoExport() const;
    void saveToSettings() const;

private:
    Ui::BatchSequenceDialog *ui;

    const int d_configureCode = 23;
    const int d_quickCode = 27;
};

#endif // BATCHSEQUENCEDIALOG_H
