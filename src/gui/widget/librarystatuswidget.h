#ifndef LIBRARYSTATUSWIDGET_H
#define LIBRARYSTATUSWIDGET_H

#include <QWidget>

class QTableWidget;
class QTableWidgetItem;
class QTextEdit;
class QLineEdit;
class QCheckBox;
class QPushButton;
class QLabel;
class VendorLibrary;

class LibraryStatusWidget : public QWidget
{
    Q_OBJECT
public:
    explicit LibraryStatusWidget(QWidget *parent = nullptr);

    bool hasUnstagedChanges() const;
    void revertAllChanges();

signals:
    void stagingStateChanged(bool hasChanges);

private:
    void refreshLibraryStatus();
    void onLibrarySelectionChanged(QTableWidgetItem *current, QTableWidgetItem *previous);
    void onLibraryPathChanged();
    void onBrowseLibraryPath();
    void onTestLoadLibrary();
    void updateLibraryDetails(VendorLibrary &library);
    void updateLibraryConfiguration(VendorLibrary &library);
    QString getLibraryStatusText(VendorLibrary &library) const;
    QString getLibraryDisplayName(VendorLibrary &library) const;
    QString getLibraryVersion(VendorLibrary &library) const;
    void updateStagingIndicators();
    void updateControlStagingIndicator(QWidget *control, bool isModified);
    void updateAllStagingIndicators();
    QString getGenericInstallationGuidance() const;

    QTableWidget *p_libraryOverviewTable;
    QTextEdit *p_libraryDetailsText;
    QLineEdit *p_userLibraryPathEdit;
    QLineEdit *p_additionalPathsEdit;
    QCheckBox *p_autoDiscoveryCheckBox;
    QPushButton *p_browseLibraryButton;
    QPushButton *p_testLoadButton;
    QPushButton *p_refreshLibraryButton;
    QLabel *p_libraryConfigPanelLabel;
    QTextEdit *p_installationGuidanceText;

    QString d_currentLibraryKey;
    VendorLibrary *p_currentLibrary;
};

#endif // LIBRARYSTATUSWIDGET_H
