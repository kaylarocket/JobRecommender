import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../models/user.dart';
import '../../providers/auth_provider.dart';
import '../role_selection_page.dart';

class CompanyProfilePage extends StatefulWidget {
  const CompanyProfilePage({super.key});

  @override
  State<CompanyProfilePage> createState() => _CompanyProfilePageState();
}

class _CompanyProfilePageState extends State<CompanyProfilePage> {
  bool _isEditing = false;
  final _contactNameCtrl = TextEditingController();
  final _companyNameCtrl = TextEditingController();
  final _companyLocationCtrl = TextEditingController();
  bool _didInit = false;

  @override
  void dispose() {
    _contactNameCtrl.dispose();
    _companyNameCtrl.dispose();
    _companyLocationCtrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final auth = context.watch<AuthProvider>();
    final profile = auth.session?.profile;

    if (!_didInit && profile != null) {
      _populateControllers(profile);
      _didInit = true;
    }

    final companyNameDisplay =
        _displayValue(profile?.companyName, fallback: 'Your company');
    final companyNameDetail =
        _displayValue(profile?.companyName, fallback: 'Not set');
    final companyLocation =
        _displayValue(profile?.companyLocation, fallback: 'Not set');
    final companyLocationDisplay =
        _displayValue(profile?.companyLocation, fallback: 'Location not set');
    final contactName =
        _displayValue(profile?.fullName, fallback: 'Not set');
    final email =
        _displayValue(profile?.email, fallback: 'Not set');
    return Column(
      children: [
        // Title
        Padding(
          padding: const EdgeInsets.all(16),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Text(
                'Company Profile',
                style: TextStyle(fontSize: 24, fontWeight: FontWeight.w800),
              ),
              if (!_isEditing)
                TextButton.icon(
                  onPressed: profile == null
                      ? null
                      : () {
                          _populateControllers(profile);
                          setState(() => _isEditing = true);
                        },
                  icon: const Icon(Icons.edit_outlined),
                  label: const Text('Edit'),
                )
              else
                TextButton(
                  onPressed: auth.isLoading
                      ? null
                      : () => _saveProfile(context, auth),
                  child: const Text('Save'),
                ),
            ],
          ),
        ),
        // Content
        Expanded(
          child: ListView(
            padding: const EdgeInsets.all(16),
            children: [
              Container(
                padding: const EdgeInsets.all(20),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(16),
                  border: Border.all(color: const Color(0xFFE2E8F0)),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.center,
                  children: [
                    Container(
                      width: 80,
                      height: 80,
                      decoration: BoxDecoration(
                        color: const Color(0xFFEEF2FF),
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: const Center(
                        child: Text('🏢', style: TextStyle(fontSize: 48)),
                      ),
                    ),
                    const SizedBox(height: 16),
                    Text(
                      companyNameDisplay,
                      style: const TextStyle(
                        fontSize: 20,
                        fontWeight: FontWeight.w700,
                      ),
                      textAlign: TextAlign.center,
                    ),
                    const SizedBox(height: 6),
                    Text(
                      companyLocationDisplay,
                      style: const TextStyle(
                        fontSize: 12,
                        color: Color(0xFF4F46E5),
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                    const SizedBox(height: 6),
                    Text(
                      contactName,
                      style: const TextStyle(
                        fontSize: 12,
                        color: Colors.black54,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 20),
              const Text(
                'Company Details',
                style: TextStyle(fontSize: 16, fontWeight: FontWeight.w700),
              ),
              const SizedBox(height: 12),
              if (_isEditing) ...[
                _formItem('Contact name', _contactNameCtrl),
                const SizedBox(height: 12),
                _formItem('Company name', _companyNameCtrl),
                const SizedBox(height: 12),
                _formItem('Company location (Country/City)', _companyLocationCtrl),
                const SizedBox(height: 12),
                _detailItem('Email', email),
              ] else ...[
                _detailItem('Contact name', contactName),
                const SizedBox(height: 12),
                _detailItem('Email', email),
                const SizedBox(height: 12),
                _detailItem('Company name', companyNameDetail),
                const SizedBox(height: 12),
                _detailItem('Company location', companyLocation),
              ],
              const SizedBox(height: 24),
              SizedBox(
                width: double.infinity,
                child: ElevatedButton.icon(
                  onPressed: () => _logout(context),
                  icon: const Icon(Icons.logout_outlined),
                  label: const Text('Logout'),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.red.shade50,
                    foregroundColor: Colors.red.shade600,
                    side: BorderSide(color: Colors.red.shade200),
                    padding: const EdgeInsets.symmetric(vertical: 12),
                  ),
                ),
              ),
              const SizedBox(height: 12),
              SizedBox(
                width: double.infinity,
                child: ElevatedButton.icon(
                  onPressed: auth.isLoading ? null : () => _confirmDeleteAccount(context, auth),
                  icon: const Icon(Icons.delete_outline),
                  label: const Text('Delete account'),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.red.shade600,
                    foregroundColor: Colors.white,
                    padding: const EdgeInsets.symmetric(vertical: 12),
                  ),
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }

  void _populateControllers(UserProfile profile) {
    _contactNameCtrl.text = profile.fullName;
    _companyNameCtrl.text = profile.companyName ?? '';
    _companyLocationCtrl.text = profile.companyLocation ?? '';
  }

  String _displayValue(String? value, {required String fallback}) {
    final trimmed = value?.trim();
    if (trimmed == null || trimmed.isEmpty) {
      return fallback;
    }
    return trimmed;
  }

  Future<void> _saveProfile(BuildContext context, AuthProvider auth) async {
    await auth.updateRecruiterProfile(
      fullName: _contactNameCtrl.text,
      companyName: _companyNameCtrl.text,
      companyLocation: _companyLocationCtrl.text,
    );
    if (!context.mounted) return;
    if (auth.error != null) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text(auth.error!)),
      );
      return;
    }
    setState(() => _isEditing = false);
  }

  Widget _formItem(String label, TextEditingController controller) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Padding(
          padding: const EdgeInsets.only(left: 4, bottom: 6),
          child: Text(
            label,
            style: const TextStyle(fontWeight: FontWeight.w700, fontSize: 12),
          ),
        ),
        TextField(
          controller: controller,
          decoration: const InputDecoration(),
        ),
      ],
    );
  }

  void _logout(BuildContext context) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Logout'),
        content: const Text('Are you sure you want to logout?'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () {
              Navigator.pop(context);
              context.read<AuthProvider>().logout();
              Navigator.of(context).pushAndRemoveUntil(
                MaterialPageRoute(builder: (_) => const RoleSelectionPage()),
                (route) => false,
              );
            },
            child: const Text('Logout', style: TextStyle(color: Colors.red)),
          ),
        ],
      ),
    );
  }

  void _confirmDeleteAccount(BuildContext context, AuthProvider auth) {
    final parentContext = context;
    showDialog(
      context: context,
      builder: (dialogContext) => AlertDialog(
        title: const Text('Delete account?'),
        content: const Text('This will permanently delete your account and all jobs you posted.'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(dialogContext),
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () async {
              Navigator.pop(dialogContext);
              await auth.deleteAccount();
              if (!parentContext.mounted) return;
              if (auth.error != null) {
                ScaffoldMessenger.of(parentContext).showSnackBar(
                  SnackBar(content: Text(auth.error!)),
                );
                return;
              }
              Navigator.of(parentContext).pushAndRemoveUntil(
                MaterialPageRoute(builder: (_) => const RoleSelectionPage()),
                (route) => false,
              );
            },
            style: TextButton.styleFrom(foregroundColor: Colors.red),
            child: const Text('Delete'),
          ),
        ],
      ),
    );
  }

  Widget _detailItem(String label, String value) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: const Color(0xFFE2E8F0)),
      ),
      child: Row(
        children: [
          const Icon(Icons.info_outlined, color: Color(0xFF4F46E5), size: 20),
          const SizedBox(width: 12),
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                label,
                style: const TextStyle(
                  fontSize: 12,
                  color: Colors.black54,
                  fontWeight: FontWeight.w500,
                ),
              ),
              const SizedBox(height: 2),
              Text(
                value,
                style:
                    const TextStyle(fontSize: 14, fontWeight: FontWeight.w600),
              ),
            ],
          ),
        ],
      ),
    );
  }
}
