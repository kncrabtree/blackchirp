#ifndef BATCHSEQUENCEDIALOG_H
#define BATCHSEQUENCEDIALOG_H

#include <QDialog>
#include <src/data/storage/settingsstorage.h>

namespace Ui {
class BatchSequenceDialog;
}

namespace BC {
namespace Key {
static const QString batchExperiments("numExpts");
static const QString batchInterval("interval");
static const QString batchAutoExport("autoExport");
}
}

class BatchSequenceDialog : public QDialog, public SettingsStorage
{
    Q_OBJECT

public:
    explicit BatchSequenceDialog(QWidget *parent = 0);
    ~BatchSequenceDialog();

    void setQuickExptEnabled(bool en);

    static const int configureCode = 23;
    static const int quickCode = 27;

private:
    Ui::BatchSequenceDialog *ui;


};

#endif // BATCHSEQUENCEDIALOG_H
