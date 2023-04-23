#ifndef COMMUNICATIONDIALOG_H
#define COMMUNICATIONDIALOG_H

#include <QDialog>
#include <data/storage/settingsstorage.h>

namespace Ui {
class CommunicationDialog;
}

class QLabel;

class CommunicationDialog : public QDialog
{
	Q_OBJECT
	
public:
	explicit CommunicationDialog(QWidget *parent = 0);
	~CommunicationDialog();
	
private:
	Ui::CommunicationDialog *ui;
    struct CustomInfo {
        QLabel *labelWidget;
        QWidget *displayWidget;
        QString type;
        QString key;
    };

    QList<CustomInfo> d_customInfoList;
    SettingsStorage d_storage;

	void startTest(QString type, QString key);



public slots:
	void gpibDeviceChanged(int index);
	void tcpDeviceChanged(int index);
	void rs232DeviceChanged(int index);
    void customDeviceChanged(int index);
	void testGpib();
	void testTcp();
	void testRs232();
    void testCustom();

	void testComplete(QString device, bool success, QString msg);

signals:
	void testConnection(QString, QString);
};

#endif // COMMUNICATIONDIALOG_H
