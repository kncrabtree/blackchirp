#ifndef COMMUNICATIONDIALOG_H
#define COMMUNICATIONDIALOG_H

#include <QDialog>
#include <QMap>
#include <data/storage/settingsstorage.h>
#include <hardware/core/communication/communicationprotocol.h>

class QListWidget;
class QListWidgetItem;
class QStackedWidget;
class QComboBox;
class QSpinBox;
class QLineEdit;
class QLabel;
class QPushButton;
class QSplitter;
class QGroupBox;
class QFormLayout;
class ProtocolWidget;

/**
 * @brief Master-detail communication settings dialog
 * 
 * Left panel: List of all hardware devices with status indicators
 * Right panel: Configuration for selected device including protocol selection
 */
class CommunicationDialog : public QDialog
{
    Q_OBJECT
    
public:
    explicit CommunicationDialog(QWidget *parent = nullptr);
    ~CommunicationDialog();
    
private slots:
    void onDeviceSelectionChanged();
    void onProtocolChanged();
    void onTestDevice();
    void onTestAllDevices();
    void onProtocolSettingsChanged();
    
private:
    struct DeviceInfo {
        QString hwKey;
        QString subKey;
        QString name;
        CommunicationProtocol::CommType currentProtocol;
        QVector<CommunicationProtocol::CommType> supportedProtocols;
        bool connected = false;
        bool tested = false;
    };
    
    void setupUI();
    void setupLeftPanel();
    void setupRightPanel();
    void connectSignals();
    
    void populateDeviceList();
    void loadDeviceInfo();
    void updateDeviceListItem(const QString& hwKey);
    void updateRightPanel();
    void loadDeviceSettings();
    void saveDeviceSettings();
    void saveCommonSettings();
    void loadReadOptions(CommunicationProtocol::CommType protocolType);
    
    QString getDeviceDisplayText(const DeviceInfo& info);
    QIcon getStatusIcon(const DeviceInfo& info);
    
    // Left panel - device list
    QListWidget *p_deviceList;
    
    // Right panel - device configuration
    QGroupBox *p_deviceConfigGroup;
    QLabel *p_deviceNameLabel;
    QComboBox *p_protocolCombo;
    QStackedWidget *p_protocolStack;
    
    // Common settings (read options)
    QGroupBox *p_readOptionsGroup;
    QSpinBox *p_timeoutSpinBox;
    QLineEdit *p_termCharEdit;
    
    // Control buttons
    QPushButton *p_testDeviceButton;
    QPushButton *p_testAllButton;
    
    // Data
    QMap<QString, DeviceInfo> d_deviceInfo;
    QMap<QString, ProtocolWidget*> d_protocolWidgets; // Key: "deviceKey:protocolType"
    QString d_currentDeviceKey;

public slots:
    void testComplete(QString device, bool success, QString msg);

signals:
    void testConnection(QString hwKey);
};

#endif // COMMUNICATIONDIALOG_H
