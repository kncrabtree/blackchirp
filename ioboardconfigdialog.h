#ifndef IOBOARDCONFIGDIALOG_H
#define IOBOARDCONFIGDIALOG_H

#include <QDialog>

namespace Ui {
class IOBoardConfigDialog;
}

class IOBoardConfigDialog : public QDialog
{
    Q_OBJECT
    
public:
    explicit IOBoardConfigDialog(QWidget *parent = nullptr);
    ~IOBoardConfigDialog();

public slots:
    void dirtySerialNumber();
    void testConnectionCallback();
    void testComplete(QString device, bool success, QString msg);
    void accept();

signals:
    void testConnection(QString type, QString key);

    
private:
    Ui::IOBoardConfigDialog *ui;
    QString d_key;
};

#endif // IOBOARDCONFIGDIALOG_H
