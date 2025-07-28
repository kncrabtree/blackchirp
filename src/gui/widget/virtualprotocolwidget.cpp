#include <gui/widget/virtualprotocolwidget.h>
#include <gui/style/themecolors.h>

#include <QVBoxLayout>
#include <QLabel>

VirtualProtocolWidget::VirtualProtocolWidget(const QString& hwKey, QWidget *parent)
    : ProtocolWidget(hwKey, parent)
{
    setupUI();
}

CommunicationProtocol::CommType VirtualProtocolWidget::getProtocolType() const
{
    return CommunicationProtocol::Virtual;
}

void VirtualProtocolWidget::setupUI()
{
    p_layout = new QVBoxLayout(this);
    
    // Message explaining virtual protocol
    p_messageLabel = new QLabel(this);
    p_messageLabel->setText(
        "<b>Virtual Protocol</b><br><br>"
        "Virtual instruments simulate hardware without actual communication.<br>"
        "No configuration settings are available or required.<br><br>"
        "Read timeout and termination characters are not applicable "
        "since no real communication occurs."
    );
    p_messageLabel->setWordWrap(true);
    p_messageLabel->setAlignment(Qt::AlignTop | Qt::AlignLeft);
    p_messageLabel->setStyleSheet(QString(
        "QLabel { "
        "padding: 20px; "
        "color: %1; "
        "}"
    ).arg(ThemeColors::getCSSColor(ThemeColors::SubtleText, this)));
    
    p_layout->addWidget(p_messageLabel);
    p_layout->addStretch(); // Push content to top
}

void VirtualProtocolWidget::loadProtocolSettings()
{
    // Virtual protocol has no settings to load
    // This method is intentionally empty
}

void VirtualProtocolWidget::saveProtocolSpecificSettings()
{
    // Virtual protocol has no settings to save
    // This method is intentionally empty
}