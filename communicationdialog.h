#ifndef COMMUNICATIONDIALOG_H
#define COMMUNICATIONDIALOG_H

#include <QDialog>

namespace Ui {
class CommunicationDialog;
}

class CommunicationDialog : public QDialog
{
	Q_OBJECT
	
public:
	explicit CommunicationDialog(QWidget *parent = 0);
	~CommunicationDialog();
	
private:
	Ui::CommunicationDialog *ui;

	QStringList gpibDevices;
	QStringList tcpDevices;
	QStringList rs232Devices;

	void startTest(QString type, QString key);

public slots:
	void gpibDeviceChanged(int index);
	void tcpDeviceChanged(int index);
	void rs232DeviceChanged(int index);
	void testGpib();
	void testTcp();
	void testRs232();

	void testComplete(QString device, bool success, QString msg);

signals:
	void testConnection(QString, QString);
};

#endif // COMMUNICATIONDIALOG_H
