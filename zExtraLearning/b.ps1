# The Text OleDB driver is only available in PowerShell x86. Start x86 shell if using x64.
# This has to be the first check this script performs.
if ($env:Processor_Architecture -ne "x86") { 
    Write-Warning "Switching to x86 shell"
    $ps32exe = "$env:windir\syswow64\windowspowershell\v1.0\powershell.exe"
    &$ps32exe  "$PSCommandPath $args"; return
}

# Database variables
$sqlserver = "sqlserver"
$database = "locations"
$table = "allcountries"

# CSV variables
$csvfile = "C:\temp\million.csv"
$csvdelimiter = ","
$firstRowColumnNames = $true


################### No need to modify anything below ###################
Write-Host "Script started..."
$elapsed = [System.Diagnostics.Stopwatch]::StartNew() 
[void][Reflection.Assembly]::LoadWithPartialName("System.Data")
[void][Reflection.Assembly]::LoadWithPartialName("System.Data.SqlClient")

# Setup bulk copy
$connectionstring = "Data Source=$sqlserver;Integrated Security=true;Initial Catalog=$database" 
$bulkcopy = New-Object Data.SqlClient.SqlBulkCopy($connectionstring, [System.Data.SqlClient.SqlBulkCopyOptions]::TableLock)
$bulkcopy.DestinationTableName = $table
$bulkcopy.bulkcopyTimeout = 0 

# Setup OleDB using Microsoft Text Driver.
$datasource = Split-Path $csvfile
$tablename = (Split-Path $csvfile -leaf).Replace(".", "#")
switch ($firstRowColumnNames) {
    $true { $firstRowColumnNames = "Yes" }
    $false { $firstRowColumnNames = "No" }
}
$connstring = "Provider=Microsoft.Jet.OLEDB.4.0;Data Source=$datasource;Extended Properties='text;HDR=$firstRowColumnNames;FMT=Delimited($csvdelimiter)';"
$conn = New-Object System.Data.OleDb.OleDbconnection
$conn.ConnectionString = $connstring
$conn.Open()
$cmd = New-Object System.Data.OleDB.OleDBCommand
$cmd.Connection = $conn

# Perform select on CSV file, then add results to a datatable using ExecuteReader
$cmd.CommandText = "SELECT * FROM [$tablename]"
$bulkCopy.WriteToServer($cmd.ExecuteReader([System.Data.CommandBehavior]::CloseConnection))

# Get Totals
$totaltime = [math]::Round($elapsed.Elapsed.TotalSeconds, 2)

$conn.Close(); $conn.Open()
$cmd.CommandText = "SELECT count(*) FROM [$tablename]"
$totalrows = $cmd.ExecuteScalar()
Write-Host "Total Elapsed Time: $totaltime seconds. $totalrows rows added." -ForegroundColor Green

# Clean Up
$cmd.Dispose(); $conn.Dispose(); $bulkcopy.Dispose()