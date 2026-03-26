#ifndef APPLICATIONCONFIGDIALOG_H
#define APPLICATIONCONFIGDIALOG_H

#include <QDialog>
#include <QMap>
#include <QVariant>

class BCSavePathWidget;
class QDialogButtonBox;

class ApplicationConfigDialog : public QDialog
{
    Q_OBJECT
public:
    explicit ApplicationConfigDialog(bool firstRun = false, QWidget *parent = nullptr);

public slots:
    void accept() override;

private:
    BCSavePathWidget *p_savePathWidget;
    QDialogButtonBox *p_buttons;
    QMap<QString, QVariant> d_pendingChanges;
    bool d_firstRun;
};

#endif // APPLICATIONCONFIGDIALOG_H
