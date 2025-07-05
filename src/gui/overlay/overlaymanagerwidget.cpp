#include "overlaymanagerwidget.h"
#include "bcexpoverlaydialog.h"
#include <QVBoxLayout>
#include <QLabel>
#include <QDebug>
#include <QTableView>
#include <QHeaderView>
#include <gui/widget/ftmwviewwidget.h>

OverlayManagerWidget::OverlayManagerWidget(QWidget *parent, int number, const QVector<std::shared_ptr<OverlayBase>> &overlays)
    : QWidget{parent, Qt::Window}, p_plotIdDelegate(nullptr)
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
    populateWithExistingOverlays(overlays);
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
        QWidget* tabWidget = nullptr;

        switch(typeValue)
        {
        case OverlayBase::BCExperiment:
            tabName = "BC Experiments";
            tabWidget = createBCExperimentTab();
            break;
        case OverlayBase::SPCAT:
            tabName = "SPCAT Catalogs";
            tabWidget = createPlaceholderTab(tabName);
            break;
        case OverlayBase::GenericXY:
            tabName = "Generic Data";
            tabWidget = createPlaceholderTab(tabName);
            break;
        default:
            tabName = typeName; // Fallback to enum name
            tabWidget = createPlaceholderTab(tabName);
            break;
        }

        // Store the overlay type as tab data for future use
        using namespace BC::Property::Overlay;
        tabWidget->setProperty(overlayType.toLocal8Bit().constData(), static_cast<int>(typeValue));

        // Add tab to tab widget
        p_tabWidget->addTab(tabWidget, tabName);

    }
}

QWidget *OverlayManagerWidget::createBCExperimentTab()
{
    auto tabWidget = new QWidget;
    auto tabLayout = new QVBoxLayout(tabWidget);

    // Create model and table view
    p_bcExperimentModel = new BCExperimentOverlayModel(this);
    p_bcExperimentTableView = new QTableView(tabWidget);
    p_bcExperimentTableView->setModel(p_bcExperimentModel);

    // Configure table view
    p_bcExperimentTableView->setSelectionBehavior(QAbstractItemView::SelectRows);
    p_bcExperimentTableView->setSelectionMode(QAbstractItemView::ExtendedSelection);
    p_bcExperimentTableView->setAlternatingRowColors(true);
    p_bcExperimentTableView->setSortingEnabled(false); // Disable for now

    // Configure headers
    auto horizontalHeader = p_bcExperimentTableView->horizontalHeader();
    horizontalHeader->setStretchLastSection(true);
    horizontalHeader->setSectionResizeMode(QHeaderView::Interactive);

    auto verticalHeader = p_bcExperimentTableView->verticalHeader();
    verticalHeader->setDefaultSectionSize(25);
    verticalHeader->setVisible(false);

    // Set up plot ID combo box delegate
    setupPlotIdDelegate();

    tabLayout->addWidget(p_bcExperimentTableView);

    return tabWidget;
}

QWidget *OverlayManagerWidget::createPlaceholderTab(const QString &tabName)
{
    auto tabWidget = new QWidget;
    auto tabLayout = new QVBoxLayout(tabWidget);

    // Add placeholder content
    auto placeholderLabel = new QLabel(QString("Overlays of type '%1' will be managed here.\n\nImplementation coming soon...").arg(tabName));
    placeholderLabel->setAlignment(Qt::AlignCenter);
    placeholderLabel->setStyleSheet("color: gray; font-style: italic;");
    placeholderLabel->setWordWrap(true);

    tabLayout->addWidget(placeholderLabel);

    return tabWidget;
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

    // Handle different overlay types
    switch(type) {
    case OverlayBase::BCExperiment:
        {
            // Get the FtmwViewWidget parent
            FtmwViewWidget* ftmwParent = qobject_cast<FtmwViewWidget*>(parentWidget());
            if(!ftmwParent) {
                qDebug() << "Warning: OverlayManagerWidget parent is not FtmwViewWidget";
                return;
            }
            
            // Get plot names from parent and create the dialog
            QStringList plotNames = ftmwParent->getPlotNames();
            BCExpOverlayDialog dialog(plotNames, ftmwParent);
            if(dialog.exec() == QDialog::Accepted) {
                // Create the overlay
                auto overlay = dialog.createOverlay();
                if(overlay != nullptr) {
                    // Add to parent FtmwViewWidget storage
                    ftmwParent->addOverlay(overlay);
                    
                    // Add to local model for display
                    if(p_bcExperimentModel != nullptr) {
                        p_bcExperimentModel->addOverlay(overlay);
                    }
                    
                    // Emit signal for any listeners
                    emit overlayAdded(overlay);
                    
                    qDebug() << "BCExpOverlay created and added successfully";
                }
            }
            break;
        }
    case OverlayBase::SPCAT:
        qDebug() << "SPCAT overlay creation not yet implemented";
        break;
    case OverlayBase::GenericXY:
        qDebug() << "GenericXY overlay creation not yet implemented";
        break;
    default:
        qDebug() << "Unknown overlay type";
        break;
    }
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

void OverlayManagerWidget::populateWithExistingOverlays(const QVector<std::shared_ptr<OverlayBase>> &overlays)
{
    // Add existing overlays to the appropriate models
    for(const auto& overlay : overlays)
    {
        if(overlay == nullptr)
            continue;
            
        switch(overlay->type())
        {
        case OverlayBase::BCExperiment:
            if(p_bcExperimentModel != nullptr)
                p_bcExperimentModel->addOverlay(overlay);
            break;
        case OverlayBase::SPCAT:
            // TODO: Add to SPCAT model when implemented
            break;
        case OverlayBase::GenericXY:
            // TODO: Add to GenericXY model when implemented
            break;
        default:
            break;
        }
    }
}

void OverlayManagerWidget::setupPlotIdDelegate()
{
    // Get plot names from parent FtmwViewWidget
    QStringList plotNames;
    FtmwViewWidget* ftmwParent = qobject_cast<FtmwViewWidget*>(parentWidget());
    if (ftmwParent) {
        plotNames = ftmwParent->getPlotNames();
    }
    
    // Create and set the delegate for the PlotId column
    p_plotIdDelegate = new PlotIdComboBoxDelegate(plotNames, this);
    p_bcExperimentTableView->setItemDelegateForColumn(1, p_plotIdDelegate); // PlotIdColumn = 1
}
