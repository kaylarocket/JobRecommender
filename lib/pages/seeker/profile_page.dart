import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../providers/auth_provider.dart';
import '../../widgets/primary_button.dart';
import '../role_selection_page.dart';

class ProfilePage extends StatefulWidget {
  const ProfilePage({super.key});

  @override
  State<ProfilePage> createState() => _ProfilePageState();
}

class _ProfilePageState extends State<ProfilePage> {
  final locationCtrl = TextEditingController();
  final skillsCtrl = TextEditingController();
  final headlineCtrl = TextEditingController();

  @override
  void initState() {
    super.initState();
    final profile = context.read<AuthProvider>().session?.profile;
    locationCtrl.text = profile?.preferredLocation ?? '';
    skillsCtrl.text = profile?.skills ?? '';
    headlineCtrl.text = profile?.headline ?? '';
  }

  @override
  void dispose() {
    locationCtrl.dispose();
    skillsCtrl.dispose();
    headlineCtrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final profile = context.watch<AuthProvider>().session?.profile;
    if (profile == null) {
      return const Center(child: Text('Not signed in'));
    }
    return Padding(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const CircleAvatar(radius: 26, backgroundColor: Color(0xFF4F46E5), child: Icon(Icons.person, color: Colors.white)),
              const SizedBox(width: 12),
              Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                Text(profile.fullName, style: const TextStyle(fontSize: 18, fontWeight: FontWeight.w800)),
                Text(profile.email, style: const TextStyle(color: Colors.black54)),
              ])
            ],
          ),
          const SizedBox(height: 20),
          TextField(
            controller: headlineCtrl,
            decoration: const InputDecoration(labelText: 'Headline / role'),
          ),
          const SizedBox(height: 12),
          TextField(
            controller: locationCtrl,
            decoration: const InputDecoration(labelText: 'Preferred location'),
          ),
          const SizedBox(height: 12),
          TextField(
            controller: skillsCtrl,
            maxLines: 2,
            decoration: const InputDecoration(labelText: 'Key skills'),
          ),
          const SizedBox(height: 16),
          PrimaryButton(
            label: 'Update profile',
            onPressed: () {
              ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Profile updated locally. TODO: sync to API.')));
            },
          ),
          const SizedBox(height: 12),
          OutlinedButton(
            onPressed: () {
              context.read<AuthProvider>().logout();
              Navigator.pushAndRemoveUntil(
                context,
                MaterialPageRoute(builder: (_) => const RoleSelectionPage()),
                (_) => false,
              );
            },
            child: const Text('Sign out'),
          )
        ],
      ),
    );
  }
}
