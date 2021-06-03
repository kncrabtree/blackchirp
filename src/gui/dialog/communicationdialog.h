#ifndef COMMUNICATIONDIALOG_H
#define COMMUNICATIONDIALOG_H

#include <QDialog>

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

    QList<QPair<QString,QString>> d_gpibDevices, d_tcpDevices, d_rs232Devices, d_customDevices;
    QList<CustomInfo> d_customInfoList;

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
