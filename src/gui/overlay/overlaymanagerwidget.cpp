#include "overlaymanagerwidget.h"
#include <QVBoxLayout>
#include <QLabel>
#include <QDebug>

OverlayManagerWidget::OverlayManagerWidget(QWidget *parent, int number)
    : QWidget{parent, Qt::Window}
{
    // Set window attributes
    if(number > 0)
        setWindowTitle(QString("Overlay Manager: Experiment %1").arg(number));
    else
        setWindowTitle("Overlay Manager: Main Window");
    setWindowIcon(QIcon(":/icons/peak-find.svg")); // Temporary icon
    setAttribute(Qt::WA_DeleteOnClose);
    resize(600, 400);

    setupUI();
    createTabs();
    updateButtonStates();
}

void OverlayManagerWidget::setupUI()
{
    // Create main layout
    auto mainLayout = new QVBoxLayout(this);
    mainLayout->setContentsMargins(0, 0, 0, 0);
    mainLayout->setSpacing(0);

    // Create toolbar
    p_toolBar = new QToolBar(this);
    p_toolBar->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);

    // Create actions
    p_addAction = p_toolBar->addAction(QIcon(":/icons/add.png"), "Add Overlay");
    p_addAction->setToolTip("Add a new overlay to the current plot");

    p_removeAction = p_toolBar->addAction(QIcon(":/icons/remove.png"), "Remove Overlay");
    p_removeAction->setToolTip("Remove the selected overlay");

    // Add separator
    p_toolBar->addSeparator();

    // Add raise parent action
    p_raiseParentAction = p_toolBar->addAction(QIcon(":/icons/go-up.svg"), "Show Parent");
    p_raiseParentAction->setToolTip("Bring the parent window to front");

    // Add spacer
    auto spacer = new QWidget;
    spacer->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Minimum);
    p_toolBar->addWidget(spacer);

    // Create tab widget
    p_tabWidget = new QTabWidget(this);

    // Add widgets to layout
    mainLayout->addWidget(p_toolBar);
    mainLayout->addWidget(p_tabWidget);

    // Connect signals
    connect(p_addAction, &QAction::triggered, this, &OverlayManagerWidget::addOverlay);
    connect(p_removeAction, &QAction::triggered, this, &OverlayManagerWidget::removeOverlay);
    connect(p_raiseParentAction, &QAction::triggered, this, &OverlayManagerWidget::raiseParent);
    connect(p_tabWidget, &QTabWidget::currentChanged, this, &OverlayManagerWidget::onTabChanged);
}

void OverlayManagerWidget::createTabs()
{
    // Use Q_ENUM to get all overlay types
    auto metaEnum = QMetaEnum::fromType<OverlayBase::OverlayType>();

    for(int i = 0; i < metaEnum.keyCount(); ++i)
    {
        QString typeName = metaEnum.key(i);
        auto typeValue = static_cast<OverlayBase::OverlayType>(metaEnum.value(i));

        // Create friendly names for tabs
        QString tabName;
        switch(typeValue)
        {
        case OverlayBase::BCExperiment:
            tabName = "BC Experiments";
            break;
        case OverlayBase::SPCAT:
            tabName = "SPCAT Catalogs";
            break;
        case OverlayBase::GenericXY:
            tabName = "Generic Data";
            break;
        default:
            tabName = typeName; // Fallback to enum name
            break;
        }

        // Create placeholder widget for each tab
        auto tabWidget = new QWidget;
        auto tabLayout = new QVBoxLayout(tabWidget);

        // Add placeholder content
        auto placeholderLabel = new QLabel(QString("Overlays of type '%1' will be managed here.\n\nImplementation coming soon...").arg(tabName));
        placeholderLabel->setAlignment(Qt::AlignCenter);
        placeholderLabel->setStyleSheet("color: gray; font-style: italic;");
        placeholderLabel->setWordWrap(true);

        tabLayout->addWidget(placeholderLabel);

        // Add tab to tab widget
        p_tabWidget->addTab(tabWidget, tabName);

        // Store the overlay type as tab data for future use
        using namespace BC::Property::Overlay;
        tabWidget->setProperty(overlayType.toLocal8Bit().constData(), static_cast<int>(typeValue));
    }
}

void OverlayManagerWidget::onTabChanged(int index)
{
    Q_UNUSED(index)
    updateButtonStates();
}

void OverlayManagerWidget::updateButtonStates()
{
    // For now, keep buttons enabled
    // Later this will be based on current tab content and selections
    p_addAction->setEnabled(true);
    p_removeAction->setEnabled(false); // Will enable when overlays are selected
}

void OverlayManagerWidget::addOverlay()
{
    // Get current tab's overlay type
    auto currentTabWidget = p_tabWidget->currentWidget();
    if(currentTabWidget == nullptr)
        return;

    using namespace BC::Property::Overlay;
    auto type = static_cast<OverlayBase::OverlayType>(currentTabWidget->property(overlayType.toLocal8Bit().constData()).toInt());


    // For now, just debug output
    QMetaEnum metaEnum = QMetaEnum::fromType<OverlayBase::OverlayType>();
    QString typeName = metaEnum.valueToKey(static_cast<int>(type));

    qDebug() << "Add overlay of type:" << typeName;

    // TODO: Implement overlay creation dialog
    // TODO: Create actual overlay objects
    // TODO: Add to current tab's list/view
}

void OverlayManagerWidget::removeOverlay()
{
    // Get current tab
    int currentIndex = p_tabWidget->currentIndex();
    if(currentIndex < 0)
        return;

    qDebug() << "Remove overlay from current tab";

    // TODO: Implement overlay removal
    // TODO: Get selected overlay from current tab
    // TODO: Remove from plot and delete
}

void OverlayManagerWidget::raiseParent()
{
    // Get parent widget and bring it to front
    QWidget* w = parentWidget();
    if(w)
    {
        w->activateWindow();
        w->raise();
        w->show();
    }
}
