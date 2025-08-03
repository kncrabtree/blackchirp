#include <gui/widget/tcpprotocolwidget.h>
#include <hardware/core/communication/tcpinstrument.h>
#include <data/settings/hardwarekeys.h>

#include <QFormLayout>
#include <QLineEdit>
#include <QSpinBox>

TcpProtocolWidget::TcpProtocolWidget(const QString& hwKey, QWidget *parent)
    : ProtocolWidget(hwKey, parent)
{
    setupUI();
    connectSignals();
}

CommunicationProtocol::CommType TcpProtocolWidget::getProtocolType() const
{
    return CommunicationProtocol::Tcp;
}

void TcpProtocolWidget::setupUI()
{
    p_layout = new QFormLayout(this);
    
    // IP Address/Hostname
    p_ipEdit = new QLineEdit(this);
    p_ipEdit->setPlaceholderText("e.g., 192.168.1.100 or hostname");
    p_layout->addRow("IP Address:", p_ipEdit);
    
    // Port Number
    p_portSpinBox = new QSpinBox(this);
    p_portSpinBox->setRange(1, 65535);
    p_portSpinBox->setValue(5000); // Default
    p_layout->addRow("Port:", p_portSpinBox);
}

void TcpProtocolWidget::connectSignals()
{
    // Emit settingsChanged when any control changes
    connect(p_ipEdit, &QLineEdit::textChanged, this, &ProtocolWidget::settingsChanged);
    connect(p_portSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), 
            this, &ProtocolWidget::settingsChanged);
}

void TcpProtocolWidget::loadProtocolSettings()
{
    using namespace BC::Key::TCP;
    
    // Load TCP-specific settings using group-based storage with backward compatibility
    auto ipAddress = getGroupValue(BC::Key::Comm::tcp, ip, get(ip, QString("")));
    auto portNumber = getGroupValue<int>(BC::Key::Comm::tcp, port, get<int>(port, 5000));
    
    // Update UI controls
    p_ipEdit->setText(ipAddress);
    p_portSpinBox->setValue(portNumber);
}

void TcpProtocolWidget::saveProtocolSpecificSettings()
{
    using namespace BC::Key::TCP;
    
    // Save TCP-specific settings using group-based storage
    setGroupValue(BC::Key::Comm::tcp, ip, p_ipEdit->text());
    setGroupValue(BC::Key::Comm::tcp, port, p_portSpinBox->value());
}